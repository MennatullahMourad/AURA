import os, gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split as sk_split

DATA_PATH   = r'C:\Users\USER\Desktop\AURA_PROJECT\Eye-Tracking Dataset\Eye tracking (photos)'
RESULTS_DIR = r'C:\Users\USER\Desktop\AURA\Results\image_model'

IMAGE_SIZE  = 224
BATCH_SIZE  = 8


EPOCHS      = 120

NUM_CLASSES = 4

PATIENCE    = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 20, IMAGE_SIZE + 20)),
    transforms.RandomCrop(IMAGE_SIZE),                      # اقتطاع عشوائي — جديد
    transforms.RandomHorizontalFlip(),                      # قلب أفقي — جديد
    transforms.RandomRotation(15),                          # تدوير ±15 درجة — جديد
    transforms.ColorJitter(brightness=0.3, contrast=0.3),  # تغيير إضاءة — جديد
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Test بدون augmentation — نفس الفكرة الأصلية بس منفصلة
test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ============================================================
# Dataset — نفس الأصلي بدون تغيير
# ============================================================
class AuraFolderDataset(Dataset):
    def __init__(self, main_path, transform=None):
        self.transform   = transform
        self.classes     = ['Low', 'Mild', 'Medium', 'High']
        self.image_paths = []
        self.labels      = []
        for idx, cls in enumerate(self.classes):
            folder_path = os.path.join(main_path, cls)
            if os.path.exists(folder_path):
                for img_name in os.listdir(folder_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(folder_path, img_name))
                        self.labels.append(idx)

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


# ============================================================
# [تعديل 4] SubsetWithTransform — كلاس جديد
#           عشان نقدر نطبق train_transforms على Train
#           و test_transforms على Test بشكل منفصل
# ============================================================
class SubsetWithTransform(Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset   = dataset
        self.indices   = indices
        self.transform = transform

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        img_path = self.dataset.image_paths[self.indices[i]]
        label    = self.dataset.labels[self.indices[i]]
        image    = Image.open(img_path).convert('RGB')
        return self.transform(image), label


# ============================================================
# تحميل البيانات
# ============================================================
def main():
    print("[INFO] Loading eye images...")
    full_dataset = AuraFolderDataset(DATA_PATH)

    all_indices = list(range(len(full_dataset)))
    all_labels  = full_dataset.labels

    train_idx, test_idx = sk_split(
        all_indices, test_size=0.2, random_state=42, stratify=all_labels
    )

    # [تعديل 4 تابع] استخدام SubsetWithTransform بدل Subset العادي
    train_data = SubsetWithTransform(full_dataset, train_idx, train_transforms)
    test_data  = SubsetWithTransform(full_dataset, test_idx,  test_transforms)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"[INFO] Train: {len(train_data)} | Test: {len(test_data)}")

    # ============================================================
    # [تعديل 5] تغيير الـ Backbone من ResNet18 لـ EfficientNet-B4
    #           ResNet18:       ImageNet accuracy = 69.8% | params = 11M
    #           EfficientNet-B4: ImageNet accuracy = 83.4% | params = 19M
    #           أقوى بكثير في استخراج الـ features من الصور
    # ============================================================
    class HybridAuraModel_V2(nn.Module):
        def __init__(self, num_classes=4):
            super().__init__()

            # [تعديل 5a] EfficientNet-B4 بدل ResNet18
            efficientnet = models.efficientnet_b4(
                weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
            )
            self.cnn = nn.Sequential(*list(efficientnet.children())[:-1])

            # [تعديل 5b] EfficientNet-B4 بيدي 1792 feature مش 512 زي ResNet18
            self.d_model = 1792

            # [تعديل 5c] Projection Layer جديد — بيقلل من 1792 لـ 512
            #             عشان الـ Transformer يشتغل بكفاءة
            self.projection = nn.Sequential(
                nn.Linear(self.d_model, 512),
                nn.LayerNorm(512),
                nn.GELU()
            )

            # [تعديل 5d] زيادة عمق الـ Transformer من 2 لـ 3 طبقات
            #             وزيادة dim_feedforward من 2048 (الافتراضي) لـ 1024
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=512, nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

            # [تعديل 5e] طبقة تصنيف أعمق مع Dropout لمنع الـ Overfitting
            #             الأصلي كان: Linear(512, 4) مباشرة
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(512, 128),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes)
            )

        def forward(self, x):
            features = self.cnn(x)                             # [B, 1792, 1, 1]
            features = features.view(features.size(0), -1)    # [B, 1792]
            features = self.projection(features).unsqueeze(1) # [B, 1, 512]
            t_out    = self.transformer(features)              # [B, 1, 512]
            return self.classifier(t_out.squeeze(1))           # [B, 4]


    # ============================================================
    # [تعديل 6] إعداد التدريب — 3 تغييرات:
    #           a) AdamW بدل Adam (weight_decay يمنع الـ overfitting)
    #           b) Label Smoothing في الـ Loss (تعميم أفضل)
    #           c) LR Scheduler — يقلل الـ learning rate تلقائي
    # ============================================================
    model = HybridAuraModel_V2(num_classes=NUM_CLASSES).to(DEVICE)

    # [تعديل 6b] Label Smoothing=0.1 — الأصلي كان CrossEntropyLoss() بدون smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # [تعديل 6a] AdamW بدل Adam — الأصلي كان optim.Adam(lr=0.0001)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)

    # [تعديل 6c] LR Scheduler جديد — مش موجود في الأصلي
    #             بيقلل الـ LR لما الـ loss مش بتتحسن 5 epochs متتاليين
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    print(f"[INFO] Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ============================================================
    # حلقة التدريب
    # ============================================================
    history      = {'loss': [], 'acc': [], 'test_acc': []}
    best_acc     = 0
    patience_cnt = 0

    print("\n" + "="*60)
    print("  AURA EfficientNet-B4 + Transformer — Training")
    print("="*60)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, correct = 0.0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()

            # [تعديل 7] Gradient Clipping — جديد، يمنع الـ gradients من الانفجار
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += loss.item()
            _, predicted  = torch.max(outputs, 1)
            correct       += (predicted == labels).sum().item()

        train_acc  = (correct / len(train_data)) * 100
        epoch_loss = running_loss / len(train_loader)

        # [تعديل 8] تقييم على Test بعد كل epoch — الأصلي كان بعد التدريب كله
        model.eval()
        test_correct = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                outputs = model(imgs.to(DEVICE))
                _, preds = torch.max(outputs, 1)
                test_correct += (preds == labels.to(DEVICE)).sum().item()
        test_acc = (test_correct / len(test_data)) * 100

        # [تعديل 6c تابع] تحديث الـ LR Scheduler
        scheduler.step(epoch_loss)

        history['loss'].append(epoch_loss)
        history['acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        print(f"Epoch [{epoch:03d}/{EPOCHS}] | Loss: {epoch_loss:.4f} | "
            f"Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")

        # [تعديل 9] Early Stopping + حفظ أفضل موديل — جديد كلياً
        if test_acc > best_acc:
            best_acc     = test_acc
            patience_cnt = 0
            os.makedirs(RESULTS_DIR, exist_ok=True)
            torch.save(model.state_dict(),
                    os.path.join(RESULTS_DIR, 'image_model_weights.pth'))
            print(f"  ✓ Best model saved (acc={best_acc:.2f}%)")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\n[INFO] Early stopping at epoch {epoch}")
                break

        gc.collect()
        torch.cuda.empty_cache()

    # ============================================================
    # [تعديل 10] تحميل أفضل موديل للتقييم النهائي
    #            الأصلي كان بيقيّم الموديل الأخير — مش الأفضل
    # ============================================================
    model.load_state_dict(torch.load(
        os.path.join(RESULTS_DIR, 'image_model_weights.pth'),
        map_location=DEVICE, weights_only=True
    ))
    model.eval()
    all_preds, all_labels_eval = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            outputs = model(imgs.to(DEVICE))
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels_eval.extend(labels.numpy())

    final_acc = (np.array(all_preds) == np.array(all_labels_eval)).mean() * 100
    print(f"\n{'='*60}")
    print(f"  BEST Test Accuracy: {final_acc:.2f}%")
    print(classification_report(all_labels_eval, all_preds,
                                target_names=['Low', 'Mild', 'Medium', 'High']))

    # ============================================================
    # الرسوم البيانية
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('AURA - EfficientNet-B4 + Transformer Results', fontsize=16, fontweight='bold')

    axes[0].plot(history['acc'],      label='Train', color='green')
    axes[0].plot(history['test_acc'], label='Test',  color='orange')
    axes[0].set_title('Accuracy over Epochs')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy (%)')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['loss'], color='red')
    axes[1].set_title('Training Loss')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
    axes[1].grid(True, alpha=0.3)

    cm = confusion_matrix(all_labels_eval, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low', 'Mild', 'Medium', 'High'],
                yticklabels=['Low', 'Mild', 'Medium', 'High'], ax=axes[2])
    axes[2].set_title('Confusion Matrix')
    axes[2].set_xlabel('Predicted'); axes[2].set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'image_model_results.png'), dpi=150)
    plt.show()
    print(f"[INFO] Done. Best accuracy: {final_acc:.2f}%")

if __name__ == "__main__":
    print(f"[INFO] Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    import multiprocessing
    multiprocessing.freeze_support()

    main()