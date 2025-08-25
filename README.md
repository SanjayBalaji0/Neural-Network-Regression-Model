# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The objective of this project is to develop a Neural Network Regression Model that can accurately predict a target variable based on input features. The model will leverage deep learning techniques to learn intricate patterns from the dataset and provide reliable predictions.

## Neural Network Model

![Alt text](<Screenshot 2025-08-25 201721.png>)

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

## PROGRAM
### Name: S. Sanjay Balaji
### Register Number: 212223240149
```python

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(1, 16)
        self.hidden2 = nn.Linear(16, 8)
        self.output = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}
    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.output(x)
        return x

sanjay_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(sanjay_brain.parameters(), lr=0.01)

def train_model(sanjay_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        outputs = sanjay_brain(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        sanjay_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

train_model(sanjay_brain, X_train_tensor, y_train_tensor, criterion, optimizer)

```
## Dataset Information

![Alt text](<Screenshot 2025-08-25 201344.png>)

## OUTPUT

### Training Loss Vs Iteration Plot

![Alt text](<Screenshot 2025-08-25 201428.png>)

### New Sample Data Prediction

![Alt text](<Screenshot 2025-08-25 201433.png>)

## RESULT

The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.