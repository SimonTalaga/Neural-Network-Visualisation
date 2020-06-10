# Neural Network Visualisation
A functional, scalable neural network learning in real time patterns hidden in a two-dimensional space.
Some actions (such as **scaling the network** or **modifying the training data sets**) can only be done by editing the code at the moment. Though this isn't convenient, this can be done easily, since all the methods have already been implemented. The only thing missing is a GUI.

![preview](https://github.com/SimonTalaga/Neural-Network-Visualisation/blob/master/screenshots/preview.png)

## Controls 

***ACTION*** | ***EFFECT***
------------ | -------------
**Drag a unit** | Moves the unit on screen. The network can be visually reorganized this way, while keeping its wiring scheme.
**Click on a unit** | A click on a unit selects it, and has the effect to change the data draph on the right part of the screen, showing the activation values of the selected unit. 
**↑**| In the right part of the screen, switches between the different graphs available to plot. More information in the “Implemented Graphs” section.
**↓** | Same as ↑, but in the opposite direction. ← In the left part of the screen, changes to activation function of the selected unit.
**→**| Same as ←, but in the opposite direction. W Changes the type of weight initialization. Feedback at the top-left of the screen. More information in the “Different Weight Initializations” section.
**R**| Reset the weights of the edges connecting the input units to the activation unit. V Prints the current weight values and the biais.
**P** | Presents the next learning example to the network. This starts a small animation representing the feeding of the unit. This command does not make the unit learn. L Same as P, but in this case the unit will learn and correct the weights according to the error and the learning rate.
**B** | B for bulk learning. Instead of learning example by example, this command runs 𝑛 epochs (this number can be changed in the code, and is set to 10 000.
**S** | Saves the plot in an image in the plots folder, named after the current parameters.

## GenerateDataToTrain.pde

To ensure that my network was working (and it does), I created another software, the counterpart of this one, to generate data of my own that can be easily implemented in my code, formatted in the right way. This is the file **GenerateDataToTrain.pde**

## Screenshots

![1](https://github.com/SimonTalaga/Neural-Network-Visualisation/blob/master/screenshots/ex1.png)
![2](https://github.com/SimonTalaga/Neural-Network-Visualisation/blob/master/screenshots/ex1curve.png)
