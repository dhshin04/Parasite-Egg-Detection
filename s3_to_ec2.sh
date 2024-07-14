# Configurations - need to only run once
# sudo apt-get update
# sudo apt-get install awscli -y
# aws configure

S3_BUCKET="ieee-dataport"
S3_PATH_TRAIN="competition/420/Chula-ParasiteEgg-11.zip"
LOCAL_PATH_TRAIN="/home/ubuntu/data/trainingset/images/"
S3_PATH_TEST="competition/420/Chula-ParasiteEgg-11_test.zip"
LOCAL_PATH_TEST="/home/ubuntu/data/testset/images/"

aws s3 cp s3://$S3_BUCKET/$S3_PATH_TRAIN $LOCAL_PATH_TRAIN
aws s3 cp s3://$S3_BUCKET/$S3_PATH_TEST $LOCAL_PATH_TEST

# Run following commands after running shell script:
# chmod u+rx s3_to_ec2.sh (to give user read and execute permission)
# ./s3_to_ec2.sh (to execute shell script)
