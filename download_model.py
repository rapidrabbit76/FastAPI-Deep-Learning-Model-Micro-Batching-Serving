import torch
import torch.nn as nn

model = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True)
model = list(model.children())

embedding = nn.Sequential(*model[:-1], nn.Flatten()).to("cpu")
classifier = nn.Sequential(*model[-1:]).to("cpu")

jit_embedding = torch.jit.script(embedding)
jit_classifier = torch.jit.script(classifier)

torch.jit.save(jit_embedding, "model_store/embedding.zip")
torch.jit.save(jit_classifier, "model_store/classifier.zip")
