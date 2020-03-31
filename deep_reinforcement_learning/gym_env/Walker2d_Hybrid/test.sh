#!/bin/bash
set -e
project_path=$(pwd)
cd $project_path
python -m test --env_name="test"
