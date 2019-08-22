# Deploying a Machine Learning model to predict Late Loan Payers by serving it as a REST API using Docker, FLask and Kubernetes

## Steps

1. Install docker, flask and all dependencies from the requirements file
2. Train the classification model and serialize the model using pickle by executing predict-late-payers-updated.py
3. Run application.py to start the flask server and send POST requests to http://0.0.0.0:9999/api to get back predictions
4. Create a docker file indicating all the steps needed to get the app running
5. Build the docker file : **sudo docker build -t my-app .**
6. Run the docker file : **sudo docker run -p 9999:9999 my-app**
7. Send POST requests to http://0.0.0.0:9999/api using POSTMAN to get back predictions
8. Tag the container and push it to docker hub to deploy it on Kubernetes
9. Create a Google cloud Compute Engine environment and a kubernetes cluster to deploy and scale your application
10. Create a Kubernetes deployment : **kubectl create deployment paylate --image=areddy3/my-app:v1**
11. Expose our deployment : **kubectl expose deployment paylate --type=LoadBalancer --port 9999 --target-port 9999**
12. Locate the external IP for our cluster : **kubectl get service**
13. Use the external IP to send POST requests and return predictions
14. Scale your application : **kubectl scale deployment paylate --replicas=3**
15. Check your resources usage : **kubectl get deployment paylate**
15. Update your application without affecting the service : **kubectl set image deployment/paylate mle-app=areddy3/my-app:v2**


