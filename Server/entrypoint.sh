#!/bin/bash
chmod +x /wait-for-it.sh
/wait-for-it.sh db:3306 -t 10

yarn start