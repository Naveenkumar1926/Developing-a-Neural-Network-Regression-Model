# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
The objective of this experiment is to design, implement, and evaluate a Deep Learning–based Neural Network regression model to predict a continuous output variable from a given set of input features. The task is to preprocess the data, construct a neural network regression architecture, train the model using backpropagation and gradient descent, and evaluate its performance using appropriate regression metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² score.

## Neural Network Model
Include the neural network model diagram.
<img width="1095" height="745" alt="542714748-b2141eb2-ad6c-4f3d-8419-c8126c9732b7" src="https://github.com/user-attachments/assets/a10c57a4-bf02-4043-9a23-10da995e3d1c" />

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

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

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: Naveen Kumar S

### Register Number: 212224040214

```python

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


dataset = pd.read_csv('/content/sample_data/deep_learning1 - Sheet1 (1).csv')

X = dataset[['INPUT']].values
y = dataset[['OUTPUT']].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=33
)


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, 1)

        self.relu = nn.ReLU()
        self.history = {'loss': []}

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(), lr=0.001)



def train_model(model, X_train, y_train, criterion, optimizer, epochs=2000):

    for epoch in range(epochs):
        optimizer.zero_grad()

        output = model(X_train)
        loss = criterion(output, y_train)

        loss.backward()
        optimizer.step()

        model.history['loss'].append(loss.item())

        if epoch % 200 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}")



train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)



with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f"Test Loss: {test_loss.item():.6f}")



loss_df = pd.DataFrame(ai_brain.history)

loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()


X_new = torch.tensor([[9]], dtype=torch.float32)

X_new_scaled = scaler.transform(X_new)
X_new_tensor = torch.tensor(X_new_scaled, dtype=torch.float32)

prediction = ai_brain(X_new_tensor).item()

print(f"Prediction: {prediction}")


```

### Dataset Information
<img width="153" height="172" alt="Screenshot 2026-02-05 100824" src="https://github.com/user-attachments/assets/d902708d-29f1-44cd-8e90-349c47117f55" />


### OUTPUT
<img width="812" height="346" alt="Screenshot 2026-02-05 100743" src="https://github.com/user-attachments/assets/85a29a3d-bcc3-40e3-b267-d199e71b441b" />


### Training Loss Vs Iteration Plot


<img width="721" height="565" alt="Screenshot 2026-02-05 100710" src="https://github.com/user-attachments/assets/cfc19aa6-2137-491b-914b-cc7b791cf200" />


<img width="1150" height="132" alt="Screenshot 2026-02-05 100729" src="https://github.com/user-attachments/assets/6bcd4cd6-c502-4e20-9769-4cf4d24a5719" />



### New Sample Data Prediction




## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
