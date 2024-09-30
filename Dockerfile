FROM condaforge/mambaforge:24.3.0-0
LABEL maintainer="Heather Kelly <heather@slac.stanford.edu>"

#ARG PR_BRANCH=dev

ARG DESC_ENV_DIR=/opt/desc

ENV PYTHONDONTWRITEBYTECODE=1

ENV DESC_NERSC_MPI=4.0.2

# Make, which we will need for everything
RUN apt-get update -y \
    && DEBIAN_FRONTEND="noninteractive" apt-get install -y make unzip \
    && apt-get clean all

# We need a C compiler temporarily to install MPICH, which we want so that we use
# vendored MPI with conda. Not sure if we can get away with removing the gcc and g++
# compilers afterwards to save space but will try. We will use the conda gcc for everything
# else, though ideally everything would come through conda-forge.
# I have found that using the conda-forge supplied MPICH does not work
# with shifter on NERSC.
RUN apt-get update -y  \
    && DEBIAN_FRONTEND="noninteractive" apt-get install -y gcc gfortran \
    && mkdir /opt/mpich \
    && cd /opt/mpich \
    && wget http://www.mpich.org/static/downloads/${DESC_NERSC_MPI}/mpich-${DESC_NERSC_MPI}.tar.gz \
    && tar xvzf mpich-${DESC_NERSC_MPI}.tar.gz \
    && cd mpich-${DESC_NERSC_MPI} \
    && ./configure --disable-wrapper-rpath  --disable-cxx --with-device=ch3 && make -j 4 \
    && make install \
    && rm -rf /opt/mpich \
    && apt-get remove --purge -y gcc gfortran 


    #&& /sbin/ldconfig \

RUN cd /tmp  \
    && git clone https://github.com/LSSTDESC/desc-example-env  \
    && cd desc-example-env  \
    && . /opt/conda/etc/profile.d/conda.sh && conda activate base \
    && mamba install -c conda-forge -y mpich=4.2.2.*=external_* \
    && mamba env update --name base --file conda-pack.yml \
    && conda remove --force mpi4py \
    && pip install --no-binary=mpi4py --no-cache-dir mpi4py \
    && pip install --no-cache-dir "jax==0.4.23" "jaxlib[cuda12_cudnn89]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    && mamba clean --all -y \
    && pip cache purge
    
#&& CONDA_OVERRIDE_CUDA="11.8" mamba install -y "tensorflow==2.14.0=cuda118*" -c conda-forge \
    #&& mamba install -y  pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia \
    #&& mamba install -c conda-forge -y --file ./conda/condalist_gpu.txt \
   # && pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
   ## && pip install --no-cache-dir -r ./conda/piplist_gpu.txt \ 
    #&& cd .. \
