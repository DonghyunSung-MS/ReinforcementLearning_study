#!/bin/bash
set -e
project_path=$(pwd)
cd $project_path
python -m train --env_name="test"
