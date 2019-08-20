# Deploying a Machine Learning model to predit Late Loan Payers by serving it as a REST API using Docker, FLask and Kubernetes

## Steps

<p>1. Install docker, flask and all dependencies from the requirements file<p>
<p>2. Train the classification model and serialize the model using pickle by executing predict-late-payers-updated.py<p>
<p>3. Run application.py to start the flask server and send POST requests to http://0.0.0.0:9999/ to get back predictions<p>
<p>4. Create a docker file indicating all the steps needed to get the app running<p>
<p>5. Build the docker file : sudo docker build -t mle-app .<p>
<p>6. Run the docker file : sudo docker run -p 9999:9999 mle-app<p>
<p>7. Send POST requests to http://0.0.0.0:9999/ using POSTMAN to get back predictions<p>
<p>8. Tag the container and push it to docker hub to deploy it on Kubernetes<p>
<p>9. Create a Google cloud Compute Engine environment and a kubernetes cluster to deploy and scale your application<p>
<p>10. Create a Kubernetes deployment : **kubectl create deployment paylate --image=areddy3/mle-app:[version]**<p>
<p>11. Expose our deployment : **kubectl expose deployment paylate --type=LoadBalancer --port 9999 --target-port 9999**<p>
<p>12. Locate the external IP for our cluster : **kubectl get service**<p>
<p>13. Use the external IP to send POST requests and return predictions<p>
<p>14. Scale your application : **kubectl scale deployment paylate --replicas=3**<p>
<p>15. Check your resources usage : **kubectl get deployment paylate**<p>
<p>15. Update your application without affecting the service : **kubectl set image deployment/paylate mle-app=areddy3/mle-app:[new_version]**<p>


