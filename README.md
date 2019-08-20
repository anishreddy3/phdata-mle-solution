# Deploying a Machine Learning model to predit Late Loan Payers by serving it as a REST API using Docker, FLask and Kubernetes

## Steps

<p>1. Install docker<p>
<p>2. Train the classification model and serialize the model using pickle by executing predict-late-payers-updated.py<p>
<p>3. Run application.py to start the flask server and send POST requests to http://0.0.0.0:9999/ to get back predictions<p>
<p>4. Create a docker file indicating all the steps needed to get the app running<p>
<p>5. Build the docker file : sudo docker build -t mle-app .<p>
<p>6. Run the docker file : sudo docker run -p 9999:9999 mle-app<p>
<p>7. Send POST requests to http://0.0.0.0:9999/ using POSTMAN to get back predictions<p>
<p>8. Tag the container and push it to docker hub to deploy it on Kubernetes<p>


