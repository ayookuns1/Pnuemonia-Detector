


from tqdm.auto import tqdm
import torch

def train_model(model, epochs, train_dataloader, test_dataloader, loss_fn, optimizer, device):
    results = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []  
    }
   

    
    for epoch in tqdm(range(epochs)):

        model.train()

        train_loss, train_acc = 0, 0

        for batch, (X, y) in enumerate(train_dataloader):
            
            X, y = X.to(device), y.to(device)
    
            #Forward pass
            y_pred = model(X)
    
            #Calculate the loss
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()

            #zero grad
            optimizer.zero_grad()

            #loss backward()
            loss.backward()

            #optimizer step
            optimizer.step()


            y_pred_class = torch.argmax(torch.softmax(y_pred, dim = 1), dim = 1)
            train_acc += (y_pred_class == y).sum().item()/len(y_pred)

        epoch_train_loss = train_loss / len(train_dataloader)
        epoch_train_acc  = train_acc / len(train_dataloader)



        results['train_loss'].append(epoch_train_loss)
        results['train_acc'].append(epoch_train_acc)


        model.eval()

        test_loss, test_acc = 0, 0

        with torch.inference_mode():

            for batch, (X_test, y_test) in enumerate(test_dataloader):
    
                X_test, y_test = X_test.to(device), y_test.to(device)

                y_test_pred = model(X_test)

                loss = loss_fn(y_test_pred, y_test)
                test_loss += loss.item()

                test_pred_class = torch.argmax(y_test_pred, dim = 1)
                test_acc += (test_pred_class == y_test).sum().item() / len(test_pred_class)

        epoch_test_loss = test_loss / len(test_dataloader)
        epoch_test_acc = test_acc / len(test_dataloader)


        results['test_loss'].append(epoch_test_loss)
        results['test_acc'].append(epoch_test_acc)


        print(f"Epoch: {epoch + 1}/{epochs} |"
             f"Train Loss:{epoch_train_loss:.4f} | Train Accuracy : {epoch_train_acc:.4f} |"
             f"Test Loss: {epoch_test_loss:.4f} | Test Accuracy : {epoch_test_acc:.4f}")
        
        

    return results
            

    
