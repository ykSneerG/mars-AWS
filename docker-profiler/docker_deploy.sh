aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin 278507445325.dkr.ecr.eu-central-1.amazonaws.com

docker build -t docker-devicelink .
docker tag docker-devicelink:latest 278507445325.dkr.ecr.eu-central-1.amazonaws.com/docker-devicelink:latest
docker push 278507445325.dkr.ecr.eu-central-1.amazonaws.com/docker-devicelink:latest

# Update Lambda functions
# aws lambda update-function-code --function-name clrsplt-curvelink-write --image-uri 278507445325.dkr.ecr.eu-central-1.amazonaws.com/docker-devicelink:latest --region eu-central-1 >> update_lambda.log 2>&1
# aws lambda update-function-code --function-name clrsplt-swaplink-write --image-uri 278507445325.dkr.ecr.eu-central-1.amazonaws.com/docker-devicelink:latest --region eu-central-1 >> update_lambda.log 2>&1
# aws lambda update-function-code --function-name clrsplt-spyimage-read --image-uri 278507445325.dkr.ecr.eu-central-1.amazonaws.com/docker-devicelink:latest --region eu-central-1 >> update_lambda.log 2>&1
# aws lambda update-function-code --function-name clrsplt-download-link --image-uri 278507445325.dkr.ecr.eu-central-1.amazonaws.com/docker-devicelink:latest --region eu-central-1 >> update_lambda.log 2>&1

echo "Lambda-Funktionen wurden erfolgreich aktualisiert."
