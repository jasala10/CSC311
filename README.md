## CSC311 - Final Research Project

#### Towards Biologically Plausible Learning: Feedback Alignment and Convolutional Neural Networks.

Error backpropagation (BP) is a fundamental machine learning technique that has proven incredibly effective across
a wide variety of domains. While indeed suitable for training deep neural networks, it has come under scrutiny by
computational neuroscientists for its biological implausibility. Termed the weight transport problem, the requirement
for each neuron to have access to its downstream synaptic weights W^T is highly implausible for an explanatory
account of human cognition. In recent years, alternative learning algorithms without the weight transport requirement 
have been proposed. In this project, Alex and I investigate a biologically plausible alternative to the backpropagation
algorithm termed Feedback Alignment (FA), originally introduced by Lillicrap et al. Following a trial implementation 
of these algorithms, as well as a novel modification to the FA algorithm (mFA), we compare the performance
of these algorithms in the context of two distinct convolutional neural network (CNN) architectures

**Fun Fact! 🌳** My professor for PSL432 was one of the original authors of the Feedback Alignment paper alongside Timothy Lillicrap. 
We briefly spoke about this very surprising and unintuitive technique in lecture, and I was interested in exploring its behaviour further. 
My work in PSL432 laid the foundation for this project.
