FROM apache/superset:latest

USER root

RUN pip install mysqlclient

ENV ADMIN_USERNAME $ADMIN_USERNAME
ENV ADMIN_EMAIL $ADMIN_EMAIL
ENV ADMIN_PASSWORD $ADMIN_PASSWORD

COPY ./superset-init.sh /superset-init.sh
RUN chmod +x /superset-init.sh




COPY superset_config.py /app/
ENV SUPERSET_CONFIG_PATH /app/superset_config.py

COPY add_database_connections.py /app/
RUN chmod +x /app/add_database_connections.py

USER superset
ENTRYPOINT [ "/superset-init.sh" ]