FROM pgvector/pgvector:pg16

USER root

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    git \
    postgresql-server-dev-16 \
    libreadline-dev \
    zlib1g-dev \
    flex \
    bison \
    && rm -rf /var/lib/apt/lists/*

RUN cd /tmp && \
    git clone --branch master https://github.com/apache/age.git && \
    cd age && \
    make PG_CONFIG=/usr/lib/postgresql/16/bin/pg_config && \
    make PG_CONFIG=/usr/lib/postgresql/16/bin/pg_config install && \
    cd .. && \
    rm -rf age

RUN apt-get purge -y --auto-remove \
    build-essential \
    git \
    postgresql-server-dev-16 && \
    apt-get clean

RUN mkdir -p /docker-entrypoint-initdb.d

USER postgres

CMD ["postgres"]

