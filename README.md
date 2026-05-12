# VisionCare AI — Eye Disease Detection

**AIAC Fresh Graduate Assessment | Task Code: AIAC-GA-2025-01**
University of Nizwa

This project builds two deep-learning models that classify retinal/eye images into
10 disease classes. We trained both models on the same dataset so we can compare
them fairly.

The two models are:

1. **EfficientNetB0** (Keras / TensorFlow) — `VisionCare_AI_Final.ipynb`
2. **YOLOv8s-cls** (PyTorch / Ultralytics) — `VisionCare_YOLOv8_Final.ipynb`

Both notebooks follow the same pipeline: EDA → balance dataset → baseline →
k-fold CV → training → evaluation → Grad-CAM. So if you understand one, you
understand the other.


## The 10 disease classes

| # | Class (English)                       | الاسم بالعربي                  |
|---|---------------------------------------|---------------------------------|
| 0 | Healthy                               | عين سليمة                       |
| 1 | Diabetic Retinopathy                  | اعتلال الشبكية السكري           |
| 2 | Glaucoma                              | الجلوكوما (المياه الزرقاء)      |
| 3 | Myopia                                | قصر النظر                       |
| 4 | Macular Scar                          | ندبة البقعة                     |
| 5 | Central Serous Chorioretinopathy      | اعتلال المشيمية المصلي المركزي |
| 6 | Pterygium                             | الظفرة                          |
| 7 | Retinitis Pigmentosa                  | التهاب الشبكية الصباغي          |
| 8 | Retinal Detachment                    | انفصال الشبكية                  |
| 9 | Disc Edema                            | وذمة القرص البصري               |


## Project files

```
project/
├── VisionCare_AI_Final.ipynb          # EfficientNetB0 notebook
├── VisionCare_YOLOv8_Final.ipynb      # YOLOv8 notebook
├── README.md                          # this file
└── outputs/                           # plots + saved models (generated on run)
```


## What you need to run this

- A Google account (we use Google Colab + Google Drive)
- The dataset folder uploaded to your Drive at this exact path:
  `MyDrive/eys_data_project/data-2/Augmented_Dataset/`
  with one sub-folder per class (folder name = class name)
- A GPU runtime in Colab (T4 is enough — that's what we used)

> ملاحظة: المشروع كله مبني على Colab + Google Drive. لو تبي تشغّله محلياً لازم
> تعدل مسارات الـ DATA_PATH في الخلية الثالثة من كل notebook.


---

## How to run the EfficientNetB0 notebook

1. Open `VisionCare_AI_Final.ipynb` in Google Colab.
2. From the menu: **Runtime → Change runtime type → T4 GPU → Save**.
3. From the menu: **Runtime → Run all**.
4. When the first cell runs, Colab will ask you to authorize Drive access.
   Click the link, choose your account, copy the auth code, paste it back.
5. Just wait. Don't skip cells.

What happens, top to bottom:

- **Step 1–4**: mounts Drive, installs packages, checks the dataset is there.
- **Step 5**: 5 EDA plots (class distribution, sample images, pie, pixel
  intensity, imbalance ratio).
- **Step 6**: oversamples every class to 2,500 images and writes them to
  `/content/eye_balanced/`. This step is slow the first time (10–20 min).
- **Step 7**: data generators with augmentation.
- **Step 8**: DummyClassifier baseline (this is the floor we have to beat).
- **Step 9**: 5-fold cross-validation using EfficientNetB0 features + logistic
  regression. Takes a few minutes.
- **Step 10**: MLflow setup.
- **Step 11–13**: builds EfficientNetB0, then trains in two phases:
  - Phase 1: only the classifier head trains (20 epochs, frozen backbone).
  - Phase 2: unfreeze the top 50 layers and fine-tune with a tiny LR (50 epochs
    max, early stopping at patience=10).
- **Step 14–15**: confusion matrix, classification report, training curves.
- **Step 16**: model comparison table.
- **Step 17**: Grad-CAM (shows where the model looks when it makes a decision).
- **Step 18**: saves the final `.keras` model to Drive at
  `MyDrive/eys_data_project/visioncare_efficientnet.keras` and offers a
  download link.

Total runtime on T4: about **1.5–2.5 hours** for a fresh full run. Most of that
is the oversampling step (one-time) and phase-2 fine-tuning.


## How to run the YOLOv8 notebook

Same idea, different model.

1. Open `VisionCare_YOLOv8_Final.ipynb` in Colab.
2. Runtime → Change runtime type → **T4 GPU**.
3. Runtime → Run all.
4. Authorize Drive when prompted.

Things that are different from the EfficientNet flow:

- After oversampling, YOLO needs a special folder layout:
  `dataset/train/Class/img.jpg`, `dataset/val/...`, `dataset/test/...`.
  This is done automatically in Step 7.
- We compute class weights and inject them into the loss using a custom
  `WeightedClassificationTrainer` subclass (Step 12). This is YOLO-specific
  and forces the model to care more about minority classes.
- YOLO's built-in MLflow auto-logger is disabled in Step 10. **Do not turn
  it back on** — it conflicts with our own MLflow calls and crashes training.
  (Took me a full crashed run to figure that out.)

Total runtime on T4: about **1.5–2 hours** for 50 epochs, usually less because
of early stopping (patience=15).

The best weights are saved to:
- Locally: `/content/yolo_runs/visioncare_yolov8/weights/best.pt`
- On Drive: `MyDrive/eys_data_project/visioncare_yolov8_best.pt`


## Common problems / مشاكل شائعة

**"FileNotFoundError: Dataset not found"**
→ Check that Drive is actually mounted (look at the file browser on the left
side of Colab — there should be a `drive/` folder). If not, re-run Step 1.
Also double-check the path in Step 3 matches your real Drive structure.

**Colab disconnects mid-training**
→ Free Colab cuts you off after ~12 hours of activity, or earlier if you go
idle. Solutions: leave the tab active, or buy Colab Pro. If it happens, the
best weights are checkpointed every 5 epochs so you don't lose everything —
just re-run from Step 12/13 with `model = YOLO('best.pt')` to continue.

**"CUDA out of memory"**
→ Lower `BATCH_SIZE` to 16 (or 8) in Step 3 and re-run.

**YOLO MLflow error "Run ... not found"**
→ This means YOLO's internal MLflow integration got re-enabled somehow. Make
sure Step 10 ran successfully and printed `YOLO built-in MLflow logger: DISABLED`.

**Oversampling step is super slow**
→ Normal. It only runs once. The augmented images are cached at
`/content/eye_balanced/` for the rest of the session.


---

## How to use the model on a new image (inference)

After training, both notebooks have a helper function that lets you predict on
a single image.

### For EfficientNetB0:

```python
# load model
import tensorflow as tf
model = tf.keras.models.load_model('/content/drive/MyDrive/eys_data_project/visioncare_efficientnet.keras')

# load + preprocess an image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

img = load_img('path/to/eye_image.jpg', target_size=(224, 224))
x = img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)

# predict
probs = model.predict(x)[0]
pred_idx = int(np.argmax(probs))
print('Prediction:', class_names[pred_idx], '({:.1%})'.format(probs[pred_idx]))
```

### For YOLOv8:

```python
from ultralytics import YOLO

model = YOLO('/content/drive/MyDrive/eys_data_project/visioncare_yolov8_best.pt')

result = model.predict('path/to/eye_image.jpg', imgsz=224)[0]
probs = result.probs.data.cpu().numpy()
pred_idx = int(probs.argmax())
print('Prediction:', classes[pred_idx], '({:.1%})'.format(probs[pred_idx]))
```

Or just call the built-in helper `predict_eye_image(...)` from Step 18 of the
YOLO notebook — it prints the top-5 predictions and shows a confidence chart.


## How to run the Gradio web app

The YOLO notebook has a small Gradio app you can launch to test the model
through a web UI (upload an image → get a diagnosis).

1. Finish running the whole YOLO notebook (or at least up to Step 14, so
   `best.pt` exists).
2. Run the Gradio cell. After a few seconds you'll see two URLs printed:
   - A local one (`http://127.0.0.1:7860`) — only works inside Colab
   - A public one (`https://xxxxx.gradio.live`) — works from your phone or
     anyone you share it with. Lives for 72 hours.
3. Open either link in your browser.
4. Drag an eye image into the upload box (or click to browse).
5. Wait ~1 second. The model returns the top-5 predictions with confidence
   bars. Labels are in Arabic in the UI.

> ملاحظة: الرابط العام (gradio.live) يموت بعد 72 ساعة من تشغيله. لو تبيه يستمر
> أكثر، لازم تنشره على Hugging Face Spaces أو سيرفر خاص.

If you want to launch the app **without** running the whole notebook from
scratch (just the app), open a new Colab notebook and put this in one cell:

```python
!pip install -q ultralytics gradio
from google.colab import drive
drive.mount('/content/drive')

import gradio as gr
from PIL import Image
import numpy as np
from ultralytics import YOLO

# load the model you saved earlier
model = YOLO('/content/drive/MyDrive/eys_data_project/visioncare_yolov8_best.pt')

CLASSES = ['Healthy', 'Diabetic Retinopathy', 'Glaucoma', 'Myopia',
           'Macular Scar', 'Central Serous Chorioretinopathy', 'Pterygium',
           'Retinitis Pigmentosa', 'Retinal Detachment', 'Disc Edema']

def predict(image):
    img = Image.fromarray(image).convert('RGB').resize((224, 224))
    img.save('/tmp/in.jpg')
    result = model.predict('/tmp/in.jpg', verbose=False, imgsz=224)[0]
    probs = result.probs.data.cpu().numpy()
    return {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}

gr.Interface(
    fn=predict,
    inputs=gr.Image(label='Upload eye image'),
    outputs=gr.Label(num_top_classes=5, label='Diagnosis'),
    title='VisionCare AI — Eye Disease Classifier',
).launch(share=True)
```

This way you don't have to retrain anything — just load the saved weights.


---

## How to upload your own dataset

If you want to retrain on your own images:

1. Organize the data in folders, one folder per class:
   ```
   MyDataset/
     Healthy/
       img1.jpg
       img2.jpg
       ...
     Glaucoma/
       img1.jpg
       ...
     (and so on for all 10 classes)
   ```
2. Upload `MyDataset` to your Google Drive.
3. In Step 3 of either notebook, change `DATA_PATH` to point to your folder:
   ```python
   DATA_PATH = Path('/content/drive/MyDrive/path/to/MyDataset')
   ```
4. The folder names you use **must** match what the notebook expects, or you'll
   need to also update the `classes` list. Easier to rename your folders.
5. Run all.


## How the data analysis works (Step 5 explained)

The EDA section produces 5 visualizations. Here's what each one is for and
what to look for:

1. **Class distribution bar chart** — shows how many images we have per class.
   Red bars mean "barely any data here, the model will struggle". For us,
   Pterygium had only ~100 images while Healthy had thousands.

2. **Sample images grid** — one random image per class. Useful to eyeball the
   data and catch problems like "wait, why is the Glaucoma folder full of MRI
   scans". Always look at your data before training.

3. **Pie chart** — same info as the bar chart but as percentages. Tells you
   at a glance if the dataset is balanced or not. Ours wasn't.

4. **Pixel intensity distribution** — overlaid histograms of pixel values per
   class. If two classes have very different distributions, the model might be
   learning the brightness instead of the actual disease. If they're all on
   top of each other, that's good — the model has to learn real features.

5. **Imbalance ratio chart** — `max_class_count / each_class_count`. Anything
   above 3 is moderate imbalance, above 10 is severe. Pterygium showed up at
   ~50x, which is why oversampling was necessary.

The combination tells you whether you need balancing (yes), class weights
(yes), and whether your data is even usable at all (yes — same intensity
distributions = no easy shortcuts for the model).


## Expected results

These are the numbers we got on our test split. Yours might differ slightly
because of random seeds, augmentation, and how Colab schedules GPU time.

| Model              | Accuracy | Precision | Recall | F1   | AUC-ROC |
|--------------------|----------|-----------|--------|------|---------|
| Baseline (Dummy)   | ~0.10    | 0.00      | 0.00   | 0.02 | N/A     |
| Custom CNN         | 0.72     | 0.71      | 0.72   | 0.71 | 0.94    |
| EfficientNetB0     | ~0.88    | 0.87      | 0.88   | 0.87 | 0.98    |
| YOLOv8s-cls (Ours) | ~0.89    | 0.88      | 0.89   | 0.88 | 0.98    |

5-fold cross-validation mean accuracy: roughly **0.86 ± 0.02** for both
models.

> The Baseline number is whatever the most-frequent class is divided by total
> — basically 1/N if balanced, higher if imbalanced. Both real models destroy
> it, which is the whole point of having a baseline.


## What's saved after running

Both notebooks save their outputs to your Drive at
`MyDrive/eys_data_project/`:

- `visioncare_efficientnet.keras` — full EfficientNet model
- `visioncare_yolov8_best.pt` — YOLOv8 best weights
- `yolo_outputs/` — folder with all YOLO plots
- The EfficientNet notebook also saves plots to `/content/visioncare_outputs/`
  (this gets wiped when the Colab session ends, so download them).

You can load any of these later without retraining.


## License / Notes

This project was built for AIAC-GA-2025-01 (Eye Disease Detection). The
dataset is the augmented version provided in the task brief. The Grad-CAM
visualisations and Gradio app are bonus features beyond the assessment's
minimum requirements.

**This is a learning project. Do not use it for actual medical diagnosis.**
A real diagnosis needs a licensed ophthalmologist looking at a high-quality
image with proper equipment. The model is good enough to flag a "this needs
a doctor's attention" signal — nothing more.

> هذا المشروع تعليمي. لا تستخدمه للتشخيص الطبي الفعلي. النموذج فقط يقدر يشير
> إلى احتمال وجود حالة، لكن التشخيص النهائي لازم يكون من دكتور عيون مختص.


## Contact

If something doesn't work, the most likely issue is one of:
- Drive path is wrong
- Colab disconnected during training
- You ran out of GPU memory

Check the **Common problems** section above first. After that, just re-run
the broken section — most steps are idempotent (running them twice doesn't
break anything).
