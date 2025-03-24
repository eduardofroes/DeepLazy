#!/bin/bash

echo "🔐 Enter your PyPI token:"
read -s TOKEN
echo ""
twine upload --username __token__ --password "$TOKEN" dist/*
