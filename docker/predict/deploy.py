import boto3
import os

s3 = boto3.client('s3')
#s3.create_bucket(Bucket='vl.flask-app',
#                 CreateBucketConfiguration={'LocationConstraint': 'us-east-2'})
#s3.upload_file('./Dockerfile', 'vl.flask-app', 'Dockerfile')
#s3.upload_file('./MLP.pt', 'vl.flask-app', 'MLP.pt')
#s3.upload_file('./model.py', 'vl.flask-app', 'model.py')
#s3.upload_file('./webapp.py', 'vl.flask-app', 'webapp.py')
#s3.upload_file('./flask-app.zip', 'vl.flask-app', 'flask-app.zip')

ebs = boto3.client('elasticbeanstalk')


#response = ebs.create_application_version(
#            ApplicationName='flask-app',
#            VersionLabel='v1',
#            SourceBundle={'S3Bucket': 'vl.flask-app',
#                            'S3Key': 'flask-app.zip'},
#            Process=True,
#            AutoCreateApplication=True)

#ebs.create_configuration_template(
#        ApplicationName='flask-app',
#        TemplateName='templ-v1',
#        SolutionStackName='64bit Amazon Linux 2018.03 v2.15.2 running Docker 19.03.6-ce')

ebs.create_environment(
            ApplicationName='flask-app',
            EnvironmentName='flask-env-2',
            VersionLabel='v1',
            TemplateName='templ-v1',
            OptionSettings=[{'OptionName': 'IamInstanceProfile',
                             'Value': 'aws-elasticbeanstalk-ec2-role',
                             'Namespace': 'aws:autoscaling:launchconfiguration'}])

