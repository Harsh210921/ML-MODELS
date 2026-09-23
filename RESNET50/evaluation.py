

model.eval()
correct = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        preds = outputs.argmax(dim=1)
        # Check each item in the batch individually
        for i in range(len(labels)):
            if preds[i] == labels[i]:
                correct += 1

accuracy = correct / len(test_loader.dataset)
print(f"Accuracy: {accuracy * 100:.2f}%")
