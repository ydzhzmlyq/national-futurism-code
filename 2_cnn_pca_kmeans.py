import torch, torchvision.transforms as T, numpy as np
from torchvision.models import resnet50
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from glob import glob
import os, pickle, cv2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = resnet50(pretrained=True)
model.fc = torch.nn.Identity()
model.eval().to(device)

transform = T.Compose([
    T.ToPILImage(),
    T.Resize(256), T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def feat_one(img_path):
    img = cv2.imread(img_path)[:,:,::-1]
    x   = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(x).cpu().numpy()[0]

# 逐影片提取
root = 'frames'
all_feat, meta = [], []
for movie in os.listdir(root):
    paths = sorted(glob(f'{root}/{movie}/*.jpg'))
    feats = np.vstack([feat_one(p) for p in paths])
    all_feat.append(feats)
    meta.extend([(movie, os.path.basename(p).split('.')[0]) for p in paths])
all_feat = np.vstack(all_feat)

# PCA & KMeans
pca = PCA(n_components=0.95, random_state=42)
feat_pca = pca.fit_transform(all_feat)
kmeans = KMeans(n_clusters=4, n_init=50, random_state=42).fit(feat_pca)
pickle.dump({'pca':pca,'kmeans':kmeans,'meta':meta,'feat':feat_pca},
            open('cnn_pca_kmeans.pkl','wb'))
print('✅ CNN+PCA+KMeans 完成')