aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin 278507445325.dkr.ecr.eu-central-1.amazonaws.com

docker build -t docker-colorpy .
docker tag docker-colorpy:latest 278507445325.dkr.ecr.eu-central-1.amazonaws.com/docker-colorpy:latest
docker push 278507445325.dkr.ecr.eu-central-1.amazonaws.com/docker-colorpy:latest

# Update Lambda functions
aws lambda update-function-code --function-name mars-colorpy-predict-linearization --image-uri 278507445325.dkr.ecr.eu-central-1.amazonaws.com/docker-colorpy:latest --region eu-central-1 >> update_lambda.log 2>&1

echo "Lambda-Funktionen wurden erfolgreich aktualisiert."
