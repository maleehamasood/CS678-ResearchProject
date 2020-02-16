# CS678-ResearchProject
CS678: Topics in Internet Research course (LUMS) Research Project

## Project Proposal: Federated Learning in Developing Countries

### Introduction

With the introduction of several data privacy regulations and laws, for instance the GDPR, the topic of secure and private AI has been getting much thought from researchers around the world. However, one main problem that inhibits any advancement in this field is the fact that all traditional machine learning models require large amounts of data, borrowed from its users, for training and validation at the same place and allowed to be accessed by a third party. Thus, to stay in accordance with emerging online privacy laws, we need a way of obtaining user data distributed across the world, without actually ‘obtaining’ it in order to maintain user privacy. This is where federated learning comes in.  
  
Federated Learning (FL) is a distributed machine learning technique of training a centralized model using decentralized data. The learning involves many clients, that can be mobile or other portable devices, individually training a copy of the current global model that they have received from the central server, thus making sure the data never leaves the client devices. One the individual model has been built, users send only the parameters (e.g. weights) to the central server that then consolidates numerous parameters, that it has received from the numerous individual users, to update its version of the global model. The central server then sends the updated global model to the users for inference.

