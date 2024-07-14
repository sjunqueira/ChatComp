# Use AWS base Image for Python 3.12.4
FROM public.ecr.aws/lambda/python:3.12

# Install build-essential compiler and tools
RUN microdnf update -y && microdnf install -y gcc-c++ make

# Copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install packages
RUN pip install -r requirements.txt

# Copy Function Code
COPY travel_agent.py ${LAMBDA_TASK_ROOT}

# Set Permissions to make executable file
RUN chmod +x travel_agent.py

# Set CMD to your handler
CMD [ "travel_agent.lambdaHandler" ]