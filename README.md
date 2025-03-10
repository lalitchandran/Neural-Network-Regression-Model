# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The objective of this project is to develop a Neural Network Regression Model that can accurately predict a target variable based on input features. The model will leverage deep learning techniques to learn intricate patterns from the dataset and provide reliable predictions.

## Neural Network Model

![Screenshot 2025-03-10 090158](https://github.com/user-attachments/assets/2c550f33-f43b-4e3c-aa4d-10aeed597a20)



## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

# Name:S LALIT CHANDRAN
# Register Number:212223240077
```
class NeuralNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1=nn.Linear(1,10) # Changed nn.linear to nn.Linear
    self.fc2=nn.Linear(10,12) # Changed nn.linear to nn.Linear
    self.fc3=nn.Linear(12,1) # Changed nn.linear to nn.Linear
    self.relu=nn.ReLU()
    self.history={'loss':[]}
  def forward(self, x):
    x = self. relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.fc3(x) # No activation here since it's a regression task
    return x

# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet ()
criterion = nn.MSELoss ()
optimizer = optim.RMSprop (ai_brain. parameters(), lr=0.001)


def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad() # Corrected indentation and added . before zero_grad
        loss = criterion(ai_brain(X_train), y_train)
        loss.backward()
        optimizer.step()
        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
```
            
##Dataset Information
![Screenshot 2025-03-10 090641](https://github.com/user-attachments/assets/82a27bae-0c7a-49d8-92f3-62b2d4a20556)



## OUTPUT

### Training Loss Vs Iteration Plot

![Screenshot 2025-03-10 090727](https://github.com/user-attachments/assets/6928291a-f70a-400b-8a36-20dcdc74053c)




### New Sample Data Prediction

![Screenshot 2025-03-10 091133](https://github.com/user-attachments/assets/a50bef1d-6b64-4df2-9d21-a20f6b493840)



## RESULT
The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
