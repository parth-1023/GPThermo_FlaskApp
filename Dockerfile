FROM public.ecr.aws/lambda/python:3.12

# Use headless Agg so matplotlib never tries to load libGL
ENV MPLBACKEND=Agg

WORKDIR /var/task

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

COPY . .

ENV AWS_LAMBDA_FUNCTION_HANDLER=lambda_function.handler
CMD [ "lambda_function.handler" ]
