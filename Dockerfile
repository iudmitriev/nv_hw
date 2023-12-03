FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
WORKDIR /repos/asr_project_template

COPY . .

RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install

RUN chmod +x scripts/download_best_model.sh

# Expose port (for some reason...)
EXPOSE 3000
