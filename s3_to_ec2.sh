# Configurations - need to run only once
# sudo apt-get update
# sudo apt-get install awscli -y
# aws configure

S3_BUCKET = "ieee-dataport"
S3_PATH = "competition/420/Chula-ParasiteEgg-11.zip"
LOCAL_PATH = ""

aws s3 cp s3://$S3_BUCKET/$S3_PATH

# Run following commands after running shell script:
# chmod u+rx s3_to_ec2.sh (to give user read and execute permission)
# ./s3_to_ec2.sh (to execute shell script)
