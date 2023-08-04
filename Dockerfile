# Define base image/operating system
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install software
#RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl ca-certificates
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates
RUN curl -O https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.2-linux-x86_64.tar.gz 
RUN tar xvfz julia-1.9.2-linux-x86_64.tar.gz
RUN rm -f julia-1.9.2-linux-x86_64.tar.gz

# Copy files and directory structure to working directory
COPY . . 
#COPY bashrc ~/.bashrc

SHELL ["/bin/bash", "--login", "-c"]
ENV PATH=/julia-1.9.2/bin:${PATH}
RUN JULIA_PROJECT=. julia -e 'using Pkg; Pkg.instantiate()'

# Run commands specified in "run.sh" to get started

ENTRYPOINT [ "/bin/bash", "-l", "-c" ]
#ENTRYPOINT [ "/bin/bash", "/sisap23-run.sh"]
