#!/bin/bash
# Configuration
URL="https://www.philippe-fournier-viger.com/spmf/spmf.zip"
ZIP_FILE="spmf.zip"
BUILD_DIR="spmf_build"
JAR_NAME="spmf.jar"
MAIN_CLASS="ca.pfv.spmf.gui.Main"

# Exit immediately if a command exits with a non-zero status
set -e

echo "--- 1. Downloading SPMF library ---"
curl -L "$URL" -o "$ZIP_FILE"

echo "--- 2. Unpacking $ZIP_FILE ---"
mkdir -p "$BUILD_DIR"
unzip -q "$ZIP_FILE" -d "$BUILD_DIR"

echo "--- 3. Compiling source files ---"
# Find all .java files and compile them.
# The 'ca' package folder is expected at the root of the zip.
find "$BUILD_DIR" -name "*.java" > sources.txt
javac -encoding ISO-8859-1 -d "$BUILD_DIR" -cp "$BUILD_DIR" @sources.txt
rm sources.txt

echo "--- 4. Cleaning up .java files from build directory ---"
# We remove .java files so only .class and resources (icons, etc.) remain in the JAR.
find "$BUILD_DIR" -name "*.java" -delete

echo "--- 5. Creating $JAR_NAME ---"
# Creates the JAR and sets the entry point to the SPMF GUI/CLI Main class
jar cfe "$JAR_NAME" "$MAIN_CLASS" -C "$BUILD_DIR" .

echo "--- 6. Cleanup ---"
rm -rf "$BUILD_DIR" "$ZIP_FILE"

echo "------------------------------------------------"
echo "Build Successful: $JAR_NAME has been created."
echo "You can run it using: java -jar $JAR_NAME"
echo "------------------------------------------------"