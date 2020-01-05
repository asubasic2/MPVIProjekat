from model.model import Cnn
from keras.optimizers import Adam

NUM_EPOCHS = 15
INIT_LR = 1e-3

model = Cnn.build(width=256, height=256, depth=3, classes=5)
opt = Adam(lr=INIT_LR, decay=INIT_LR / (NUM_EPOCHS * 0.5))
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

model.train(
    train_images="Dataset/Znakovi_cropped_train",
    val_images="Dataset/Znakovi_cropped_test",
    val_annotations="Dataset/Ann_test",
    train_annotations="Dataset/Ann_train",
    checkpoints_path="Proba1", epochs=15, validate=True
)

model.load_weights("Proba1.14")
out = model.predict_segmentation(
    inp="Slike/cvv.jpg",
    out_fname="output.png"
)
