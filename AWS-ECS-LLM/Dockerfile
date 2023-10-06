# FROM python:3.10-slim


# RUN mkdir -p /root/.streamlit
# RUN bash -c 'echo -e "\
# [general]\n\
# email = \"\"\n\
# " > /root/.streamlit/credentials.toml'
# RUN bash -c 'echo -e "\
# [server]\n\
# enableCORS = false\n\
# " > /root/.streamlit/config.toml'

# WORKDIR /app

# COPY ./requirements.txt requirements.txt
# COPY ./app.py app.py

# RUN pip install -r requirements.txt 
    
# EXPOSE 8501

# ENTRYPOINT [ "streamlit","run" ]

# CMD ["app.py"]


FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/streamlit/streamlit-example.git .

COPY ./requirements.txt requirements.txt
COPY ./app.py app.py


RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]