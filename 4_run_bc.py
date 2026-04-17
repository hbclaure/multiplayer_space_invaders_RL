import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader



bc_folder = '/Users/houstonclaure/Desktop/multiplayer_space_invaders_RL/analysis_results/behavior_cloning_splits'
checkpoint_path = '/Users/houstonclaure/Desktop/multiplayer_space_invaders_RL/testing/bc_best_model.pt'
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

#upload dataset
train_data = torch.load(os.path.join(bc_folder, "train.pt"), weights_only=False)
test_data = torch.load(os.path.join(bc_folder, "test.pt"), weights_only=False)
eval_data = torch.load(os.path.join(bc_folder, "eval.pt"), weights_only=False)

X_train = torch.tensor(train_data["observations"], dtype=torch.float32)
y_train = torch.tensor(train_data["actions"], dtype=torch.long)
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)


#eval test
eval_x= torch.tensor(eval_data["observations"], dtype=torch.float32)
eval_y = torch.tensor(eval_data["actions"], dtype=torch.long)

#test 
test_x= torch.tensor(test_data["observations"], dtype=torch.float32)
test_y= torch.tensor(test_data["actions"], dtype=torch.long)




model = nn.Sequential(
    nn.Linear(X_train.shape[1], 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 3)
)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-3)
saved_best_models =None

print_freq = 10 

best_eval_loss = float("inf")
for epoch in range(100):
    #train
    model.train()
    # Forward pass

    for x_batch, y_batch in train_loader:

        output = model(x_batch)
        loss = criterion(output, y_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % print_freq == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')
    
    #validation pass. using specific functions to avoid gradiennt updatae during training
    model.eval()
    with torch.no_grad():
        eval_output = model(eval_x)
        eval_loss = criterion(eval_output, eval_y)

        preds = eval_output.argmax(dim=1)
        acc = (preds == eval_y).float().mean()
        if (epoch+1) % print_freq == 0:
            print(f"Epoch [{epoch+1}/100], Eval Loss: {eval_loss.item():.4f}, Eval Acc: {acc.item():.4f}")
    

    if eval_loss.item() < best_eval_loss:
        best_eval_loss = eval_loss.item()
        saved_best_models = model.state_dict()
        torch.save(model.state_dict(), checkpoint_path)


#test using best eval 
correct =0
total = 0
model.load_state_dict(saved_best_models)
model.eval()

with torch.no_grad():

    test_output = model(test_x)
    test_loss = criterion(test_output, test_y)
    test_preds = test_output.argmax(dim=1)
    correct = (test_preds == test_y).sum().item()
    total = test_y.size(0)
    test_acc = correct / total


    print(f'Final Loss : {test_loss.item():.4f} | accuracy = {test_acc:.4f}')




