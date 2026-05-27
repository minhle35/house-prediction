#!/usr/bin/env bash
# Provision Azure Blob Storage for the house-prediction model registry.
#
# Prerequisites:
#   - Azure CLI installed: https://learn.microsoft.com/en-us/cli/azure/install-azure-cli
#   - Logged in: az login
#
# Usage:
#   chmod +x scripts/provision_azure.sh
#   ./scripts/provision_azure.sh
#
# After running, copy the printed connection string into backend/.env.
set -euo pipefail

# ── Config — edit these before running ───────────────────────────────────────
RESOURCE_GROUP="house-prediction-rg"
LOCATION="australiaeast"          # closest Azure region to Sydney AU
STORAGE_ACCOUNT="housepredmodels" # must be globally unique, 3–24 lowercase alphanumeric
CONTAINER="models"
# ─────────────────────────────────────────────────────────────────────────────

echo "==> Logging in (skip if already logged in)"
az account show &>/dev/null || az login

echo "==> Creating resource group: $RESOURCE_GROUP in $LOCATION"
az group create \
  --name "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --output table

echo "==> Creating storage account: $STORAGE_ACCOUNT"
az storage account create \
  --name "$STORAGE_ACCOUNT" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --sku Standard_LRS \
  --kind StorageV2 \
  --allow-blob-public-access false \
  --output table

echo "==> Creating blob container: $CONTAINER"
az storage container create \
  --name "$CONTAINER" \
  --account-name "$STORAGE_ACCOUNT" \
  --auth-mode login \
  --output table

echo ""
echo "==> Done. Copy this connection string into backend/.env as AZURE_STORAGE_CONNECTION_STRING:"
echo ""
az storage account show-connection-string \
  --name "$STORAGE_ACCOUNT" \
  --resource-group "$RESOURCE_GROUP" \
  --query connectionString \
  --output tsv
echo ""
echo "==> Set AZURE_STORAGE_CONTAINER=$CONTAINER in backend/.env"
