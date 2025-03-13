#!/bin/bash

echo "Building plugin..."

# Clean the deps
rm -rf esmfold/deps

# Remove any existing hp file
rm *.hp

# Get the current git tag (otherwise 0.0.1 will be used)
git_tag=$(git describe --tags)

if [ "$git_tag" == "" ]; then
    echo "No git tag found, using default version"
    git_tag="0.0.1"
fi

echo "Building plugin with tag: $git_tag"

# Get the OS name and adjust sed command accordingly
sed_program="sed"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    osname="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    osname="macos_arm"
    sed_program="gsed"
else
    osname="unknown"
fi

# Update the "version" field in plugin.meta
$sed_program -i 's/"version": "0.0.1"/"version": "'$git_tag'"/g' esmfold/plugin.meta


zip -r esmfold-$git_tag.hp esmfold
