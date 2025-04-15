import sagemaker
from sagemaker.pytorch import PyTorchModel
import boto3
import os
import tarfile
import shutil
import tempfile

def package_model():
    """打包模型文件"""
    try:
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        
        # 源文件路径
        model_source = os.path.join('src', 'models', 'inference.py')
        requirements_source = 'requirements.txt'
        
        # 确保源文件存在
        if not os.path.exists(model_source):
            raise FileNotFoundError(f"Model source not found at {model_source}")
        if not os.path.exists(requirements_source):
            raise FileNotFoundError(f"Requirements not found at {requirements_source}")
            
        # 创建目标目录
        code_dir = os.path.join(temp_dir, 'code')
        os.makedirs(code_dir, exist_ok=True)
        
        # 复制文件
        shutil.copy2(model_source, os.path.join(code_dir, 'inference.py'))
        shutil.copy2(requirements_source, os.path.join(code_dir, 'requirements.txt'))
        
        print(f"Copied files to code directory: {code_dir}")
        
        # 创建tar文件
        tar_path = os.path.join(temp_dir, 'model.tar.gz')
        with tarfile.open(tar_path, 'w:gz') as tar:
            # 只添加必要的文件
            tar.add(os.path.join(code_dir, 'inference.py'), arcname='code/inference.py')
            tar.add(os.path.join(code_dir, 'requirements.txt'), arcname='code/requirements.txt')
        
        print(f"Created tar file at: {tar_path}")
        return tar_path, code_dir
        
    except Exception as e:
        print(f"Error in package_model: {e}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise

def deploy_model():
    """部署模型到SageMaker"""
    temp_dir = None
    try:
        # 使用指定的角色
        role = "arn:aws:iam::xxxxxxxxxxx:role/SageMaker-ExecutionRole"
        
        # 初始化SageMaker会话
        session = boto3.Session(region_name='ap-southeast-2')
        sagemaker_session = sagemaker.Session(boto_session=session)
        
        # 打包模型
        model_path, code_dir = package_model()
        print(f"Using model package at: {model_path}")
        
        # 上传模型到S3
        bucket = sagemaker_session.default_bucket()
        model_artifact = sagemaker_session.upload_data(
            path=model_path,
            bucket=bucket,
            key_prefix='multimodal-search/model'
        )
        
        print(f"Model uploaded to: {model_artifact}")
        
        # 创建PyTorch模型
        pytorch_model = PyTorchModel(
            model_data=model_artifact,
            role=role,
            entry_point='inference.py',
            framework_version='1.12.1',
            py_version='py38',
            source_dir=code_dir,
            sagemaker_session=sagemaker_session
        )
        
        # 部署模型
        predictor = pytorch_model.deploy(
            initial_instance_count=1,
            instance_type='ml.m4.xlarge',
            endpoint_name='multimodal-search-endpoint'
        )
        
        print(f"Model deployed to endpoint: {predictor.endpoint_name}")
        return predictor.endpoint_name
        
    except Exception as e:
        print(f"Error during deployment: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # 清理临时文件
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            print("Cleaned up temporary files")

def cleanup_endpoint(endpoint_name):
    """清理endpoint以控制成本"""
    try:
        session = boto3.Session(region_name='ap-southeast-2')
        sagemaker_client = session.client('sagemaker')
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        print(f"Endpoint {endpoint_name} deleted")
    except Exception as e:
        print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    try:
        endpoint_name = deploy_model()
        
        # 等待用户确认是否删除endpoint
        input("Press Enter to delete the endpoint and cleanup resources...\n"
              "Note: endpoint will incur costs, please delete after testing")
        cleanup_endpoint(endpoint_name)
    except Exception as e:
        print(f"Error: {e}")