#Issue: AJP Connector Threads Hung in CLOSE_WAIT Status

Issue Overview
Version: 4.2.0.GA_CP02
Affects Version/s: 4.2.0.GA, 4.2.0.GA_CP01

Environment
The issue has been reproduced in the following environment:

Web Server: Apache 2.2.3 + mod_jk 1.2.21 on RHEL5 Workstation

App Server: JBoss EAP 4.2.0.GA on RHEL4

Network: Both boxes located on the same subnet

Configuration:

httpd.conf

mod_jk.conf

workers.properties

uriworkermap.properties

Reproduction Steps
To reproduce the issue, follow these steps:

Use an out-of-the-box installation of EAP 4.2.0.GA.

Configure Apache + mod_jk to front JBoss (using the configuration files listed in the environment section).

Start JBoss binding to the specific IP:

bash
run.sh -b 192.168.2.12
Run the load test using Apache Bench (ab):

bash
ab -n 100 -k -c 10 -t 30 http://localhost/jmx-console/