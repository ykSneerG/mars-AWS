aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin 278507445325.dkr.ecr.eu-central-1.amazonaws.com

docker build -t docker-colorpy .
docker tag docker-colorpy:latest 278507445325.dkr.ecr.eu-central-1.amazonaws.com/docker-colorpy:latest
docker push 278507445325.dkr.ecr.eu-central-1.amazonaws.com/docker-colorpy:latest

# Update Lambda functions
aws lambda update-function-code --function-name mars-colorpy-predict-linearization --image-uri 278507445325.dkr.ecr.eu-central-1.amazonaws.com/docker-colorpy:latest --region eu-central-1 >> update_lambda.log 2>&1
#aws lambda update-function-code --function-name mars-colorpy-predict-linearization-V2 --image-uri 278507445325.dkr.ecr.eu-central-1.amazonaws.com/docker-colorpy:latest --region eu-central-1 >> update_lambda.log 2>&1
aws lambda update-function-code --function-name mars-colorpy-predict-linearinterpolation --image-uri 278507445325.dkr.ecr.eu-central-1.amazonaws.com/docker-colorpy:latest --region eu-central-1 >> update_lambda.log 2>&1
#aws lambda update-function-code --function-name mars-colorpy-predict-linearization-V3 --image-uri 278507445325.dkr.ecr.eu-central-1.amazonaws.com/docker-colorpy:latest --region eu-central-1 >> update_lambda.log 2>&1
aws lambda update-function-code --function-name mars-colorpy-predict-1D --image-uri 278507445325.dkr.ecr.eu-central-1.amazonaws.com/docker-colorpy:latest --region eu-central-1 >> update_lambda.log 2>&1
aws lambda update-function-code --function-name mars-colorpy-predict-area-V4 --image-uri 278507445325.dkr.ecr.eu-central-1.amazonaws.com/docker-colorpy:latest --region eu-central-1 >> update_lambda.log 2>&1
aws lambda update-function-code --function-name mars-colorpy-predict-volume-V4 --image-uri 278507445325.dkr.ecr.eu-central-1.amazonaws.com/docker-colorpy:latest --region eu-central-1 >> update_lambda.log 2>&1
aws lambda update-function-code --function-name mars-colorpy-predict-ndimensional-V4 --image-uri 278507445325.dkr.ecr.eu-central-1.amazonaws.com/docker-colorpy:latest --region eu-central-1 >> update_lambda.log 2>&1
#aws lambda update-function-code --function-name mars-colorpy-files-cgats2json --image-uri 278507445325.dkr.ecr.eu-central-1.amazonaws.com/docker-colorpy:latest --region eu-central-1 >> update_lambda.log 2>&1
aws lambda update-function-code --function-name mars-colorpy-bucket-cgats2json --image-uri 278507445325.dkr.ecr.eu-central-1.amazonaws.com/docker-colorpy:latest --region eu-central-1 >> update_lambda.log 2>&1



echo "Lambda-Funktionen wurden erfolgreich aktualisiert."
