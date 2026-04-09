#!/usr/bin/env python3
"""
Download ESD (Embedded Seamless Data) tiles from iEarth platform.

Usage:
    python download_esd.py --year 2024 --output_dir ./tiles_2024 --username <user> --password <pass>
    python download_esd.py --year 2024 --output_dir ./tiles_2024 --tiles_file esd_tiles_needed.txt --resume

Credentials are read from CLI arguments or from the environment variables:
    STARCLOUD_USERNAME
    STARCLOUD_PASSWORD
"""

import argparse
import base64
import json
import os
import time
from pathlib import Path

import requests
from Crypto.Cipher import PKCS1_v1_5
from Crypto.PublicKey import RSA

# ============ Configuration ============
STARCLOUD_AUTH_URL = "https://data-starcloud.pcl.ac.cn/starcloud/api/user/authenticate"
STARCLOUD_DOWNLOAD_URL = "https://data-starcloud.pcl.ac.cn/starcloud/api/file/downloadResource"
AIFOREARTH_API = "https://data-starcloud.pcl.ac.cn/aiforearth/api"

PUBLIC_KEY = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAvrzz4DGWHc6YmK0BZ30L
MqZvWTLOsuIzPJn9LrJ++5416UwqpnnR5DxI4NOAdwwAOv7aOdiZ6ny5u8BX5pot
v+cB3evrcpw5HbxSbj1kUzfOv4VCnGSdPMRnx/i3DCaQN1ubliJrm/jfGBEVioTN
kT+iNxcZZYxazgP1PHJOpmUwu7LME+zdGSB+y0MIZasmKi6aVFBIHug83ku0lNpA
+hdWTJu+Unsl6cD58wf7fSF3zLbb9Cmy/kg+qcS0QzzBajSXh1UuRm+4KuQZfDRD
uIagICtXvrY/u2Ow3Kdw4YGqEMe+TLiuxFoCQO9smGCOi9sCFAVrC3DaGPhGYT42
2QIDAQAB
-----END PUBLIC KEY-----"""

OBJECT_KEY_TEMPLATE = "shared-dataset/SDC30_EBD/SDC30_EBD_V001/{year}/{filename}"
OBJECT_KEY_TEMPLATE_CHN = "shared-dataset/SDC30_EBD/SDC30_EBD_V001CHN/{year}/{filename}"
RESOURCE_ID = "64"


def login(session: requests.Session, username: str, password: str):
    """Login to iEarth platform and return token information."""
    key = RSA.import_key(PUBLIC_KEY)
    cipher = PKCS1_v1_5.new(key)
    payload = json.dumps({"account": username, "password": password})
    encrypted = base64.b64encode(cipher.encrypt(payload.encode())).decode()

    resp = session.post(
        STARCLOUD_AUTH_URL,
        json={"key": encrypted},
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(f"Login failed: {data}")

    token = data["data"]["token"]
    user_id = data["data"]["userId"]
    user_name = data["data"]["userName"]
    print(f"Logged in as {user_name} (id={user_id})")
    return token, user_id, user_name


def get_file_list(session: requests.Session, year: int, data_type: str = "SDC30_EBD_V001", page: int = 1, count: int = 200):
    """Get a page of available ESD tiles for a given year."""
    resp = session.post(
        f"{AIFOREARTH_API}/data/getFileListByPage",
        json={
            "params": {
                "table": "rs_sdc30_ebd2",
                "path": f"{data_type}/{year}",
                "page": page,
                "enableSpatialQuery": False,
                "count": count,
            }
        },
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", []), data.get("total", 0)


def get_all_files(session: requests.Session, year: int, data_type: str = "SDC30_EBD_V001"):
    """Get the complete list of available ESD tiles for a year."""
    all_files = []
    page = 1
    count = 500

    while True:
        files, total = get_file_list(session, year, data_type, page, count)
        all_files.extend(files)
        print(f"  Fetched page {page}: {len(files)} files (total so far: {len(all_files)}/{total})")

        if len(all_files) >= total or len(files) == 0:
            break
        page += 1
        time.sleep(0.5)

    return all_files


def get_signed_url(session: requests.Session, object_key: str, token: str, user_id: str, user_name: str) -> str:
    """Get a signed download URL for a file."""
    resp = session.post(
        STARCLOUD_DOWNLOAD_URL,
        json={
            "objectKey": object_key,
            "resourceId": RESOURCE_ID,
            "userAccount": user_name,
            "userId": str(user_id),
            "country": "",
            "resourceType": "REMOTE_SENSING",
        },
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("signedUrl", "")


def download_file(session: requests.Session, url: str, output_path: Path, expected_size: int | None = None) -> int:
    """Download a file from a signed URL."""
    resp = session.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    total_size = int(resp.headers.get("Content-Length", 0))
    if total_size == 0 and expected_size:
        total_size = expected_size

    downloaded = 0
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)

    return downloaded


def load_needed_tiles(tiles_file: str) -> set[str]:
    path = Path(tiles_file)
    if path.suffix.lower() == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("tiles", []))

    with open(path, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ESD tiles from iEarth")
    parser.add_argument("--year", type=int, default=2024, help="Year to download (2000-2024)")
    parser.add_argument("--output_dir", type=str, default="./tiles_2024", help="Output directory")
    parser.add_argument("--tiles_file", type=str, default=None, help="JSON or TXT file with needed tile ids")
    parser.add_argument("--max_tiles", type=int, default=None, help="Max number of tiles to download")
    parser.add_argument("--resume", action="store_true", help="Skip already downloaded files")
    parser.add_argument("--concurrent", type=int, default=1, help="Reserved for future use")
    parser.add_argument("--username", type=str, default=os.getenv("STARCLOUD_USERNAME"), help="StarCloud username")
    parser.add_argument("--password", type=str, default=os.getenv("STARCLOUD_PASSWORD"), help="StarCloud password")
    args = parser.parse_args()

    if not args.username or not args.password:
        raise SystemExit(
            "Missing credentials. Provide --username/--password or set STARCLOUD_USERNAME and STARCLOUD_PASSWORD."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    needed_tiles = None
    if args.tiles_file:
        needed_tiles = load_needed_tiles(args.tiles_file)
        print(f"Filtering to {len(needed_tiles)} needed tiles")

    session = requests.Session()

    print("Logging in...")
    token, user_id, user_name = login(session, args.username, args.password)

    print(f"\nFetching file list for year {args.year}...")
    all_files = get_all_files(session, args.year)
    print(f"Total available: {len(all_files)} tiles")

    if needed_tiles:
        filtered = []
        for file_entry in all_files:
            parts = file_entry["file"].replace(".tif", "").split("_")
            tile_id = parts[3] if len(parts) >= 5 else ""
            if tile_id in needed_tiles:
                filtered.append(file_entry)
        print(f"After filtering: {len(filtered)} tiles needed (of {len(needed_tiles)} requested)")
        all_files = filtered

    if args.max_tiles:
        all_files = all_files[: args.max_tiles]
        print(f"Limited to {len(all_files)} tiles")

    total_size_mb = sum(file_entry.get("size", 0) for file_entry in all_files) / 1024 / 1024
    print(f"Total download size: {total_size_mb:.1f} MB ({total_size_mb / 1024:.1f} GB)")

    print(f"\nStarting downloads to {output_dir}...")
    success = 0
    failed = 0
    skipped = 0

    for index, file_entry in enumerate(all_files, start=1):
        filename = file_entry["file"]
        expected_size = file_entry.get("size", 0)
        output_path = output_dir / filename

        if args.resume and output_path.exists():
            existing_size = output_path.stat().st_size
            if expected_size > 0 and abs(existing_size - expected_size) < 1024:
                skipped += 1
                continue

        object_key = OBJECT_KEY_TEMPLATE.format(year=args.year, filename=filename)

        try:
            try:
                signed_url = get_signed_url(session, object_key, token, user_id, user_name)
            except Exception:
                print("  Re-logging in...")
                token, user_id, user_name = login(session, args.username, args.password)
                signed_url = get_signed_url(session, object_key, token, user_id, user_name)

            if not signed_url:
                print(f"  [{index}/{len(all_files)}] {filename}: No signed URL")
                failed += 1
                continue

            downloaded = download_file(session, signed_url, output_path, expected_size)
            success += 1
            print(f"  [{index}/{len(all_files)}] {filename}: {downloaded / 1024 / 1024:.1f} MB OK")

        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                object_key_chn = OBJECT_KEY_TEMPLATE_CHN.format(
                    year=args.year,
                    filename=filename.replace("V001", "V001CHN"),
                )
                try:
                    signed_url = get_signed_url(session, object_key_chn, token, user_id, user_name)
                    downloaded = download_file(session, signed_url, output_path, expected_size)
                    success += 1
                    print(f"  [{index}/{len(all_files)}] {filename}: {downloaded / 1024 / 1024:.1f} MB (CHN) OK")
                    continue
                except Exception:
                    pass

            print(f"  [{index}/{len(all_files)}] {filename}: FAILED - {exc}")
            failed += 1
            if output_path.exists():
                output_path.unlink()

        except Exception as exc:
            print(f"  [{index}/{len(all_files)}] {filename}: FAILED - {exc}")
            failed += 1
            if output_path.exists():
                output_path.unlink()

        time.sleep(0.5)

    print(f"\n{'=' * 50}")
    print("Download complete!")
    print(f"  Success: {success}")
    print(f"  Failed:  {failed}")
    print(f"  Skipped: {skipped}")
    print(f"  Total:   {success + failed + skipped}")


if __name__ == "__main__":
    main()
