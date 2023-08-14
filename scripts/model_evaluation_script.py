def evaluate_fitting_room_model(model, test_data):
    loss, accuracy = model.evaluate(test_data)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
