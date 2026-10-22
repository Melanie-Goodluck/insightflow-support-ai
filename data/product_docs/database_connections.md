# Database connections

## PostgreSQL setup

To connect PostgreSQL, open Settings, choose Data Sources, and select Add Data
Source. Enter the database host, port, database name, username, and password.
The default PostgreSQL port is 5432. InsightFlow requires an encrypted SSL
connection.

## Connection problems

Error E221 means InsightFlow could not reach the database server. Confirm that
the host and port are correct and that the server firewall allows the
connection. Error E223 means authentication failed; check the username and
password. Database credentials are encrypted and are never displayed after
they are saved.

