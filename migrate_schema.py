#!/usr/bin/env python3
"""Migration script to clean up schema - remove workos_user_id from drafts/feedback."""

import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os

MONGODB_URI = "mongodb+srv://db:pIwDxC2x2AV0HSwc@cluster0.32ynumk.mongodb.net/?appName=Cluster0"
DATABASE_NAME = "blazel"

async def migrate():
    client = AsyncIOMotorClient(MONGODB_URI)
    db = client[DATABASE_NAME]

    # Show current state
    print("=== Current Collections ===")
    collections = await db.list_collection_names()
    for c in collections:
        count = await db[c].count_documents({})
        print(f"  {c}: {count} docs")

    # Sample documents before
    print("\n=== Sample Drafts (before) ===")
    async for d in db.drafts.find().limit(2):
        print(f"  _id: {d.get('_id')}, customer_id: {d.get('customer_id')}, workos_user_id: {d.get('workos_user_id')}")

    print("\n=== Sample Feedback (before) ===")
    async for f in db.feedback.find().limit(2):
        print(f"  _id: {f.get('_id')}, customer_id: {f.get('customer_id')}, workos_user_id: {f.get('workos_user_id')}")

    # Migration 1: Ensure customer_id exists on drafts (copy from workos_user_id if missing)
    print("\n=== Ensuring customer_id on Drafts ===")
    result = await db.drafts.update_many(
        {"customer_id": {"$exists": False}},
        [{"$set": {"customer_id": "$workos_user_id"}}]
    )
    print(f"  Added customer_id to {result.modified_count} drafts")

    # Migration 2: Ensure customer_id exists on feedback (copy from workos_user_id if missing)
    print("\n=== Ensuring customer_id on Feedback ===")
    result = await db.feedback.update_many(
        {"customer_id": {"$exists": False}},
        [{"$set": {"customer_id": "$workos_user_id"}}]
    )
    print(f"  Added customer_id to {result.modified_count} feedback records")

    # Migration 3: Remove workos_user_id from drafts (no longer needed)
    print("\n=== Removing workos_user_id from Drafts ===")
    result = await db.drafts.update_many(
        {"workos_user_id": {"$exists": True}},
        {"$unset": {"workos_user_id": ""}}
    )
    print(f"  Removed workos_user_id from {result.modified_count} drafts")

    # Migration 4: Remove workos_user_id from feedback (no longer needed)
    print("\n=== Removing workos_user_id from Feedback ===")
    result = await db.feedback.update_many(
        {"workos_user_id": {"$exists": True}},
        {"$unset": {"workos_user_id": ""}}
    )
    print(f"  Removed workos_user_id from {result.modified_count} feedback records")

    print("\n=== Migration Complete ===")

    # Verify
    print("\n=== Sample Drafts (after) ===")
    async for d in db.drafts.find().limit(2):
        print(f"  _id: {d.get('_id')}, customer_id: {d.get('customer_id')}, workos_user_id: {d.get('workos_user_id')}")

    print("\n=== Sample Feedback (after) ===")
    async for f in db.feedback.find().limit(2):
        print(f"  _id: {f.get('_id')}, customer_id: {f.get('customer_id')}, workos_user_id: {f.get('workos_user_id')}")

    client.close()

if __name__ == "__main__":
    asyncio.run(migrate())
