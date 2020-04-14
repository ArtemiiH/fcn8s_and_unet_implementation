import argparse
import asyncio
import io
import os
import time
from typing import Any, List, Tuple

import aiofiles
import numpy as np
from aiohttp import ClientSession
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.mask import decode

MAX_TRIES = 3

PERSON_MAX_AREA_THRESHOLD = 1
PERSON_MIN_AREA_THRESHOLD = 0.25
MAX_PPL_ALLOWED = 6


async def download_image(session: ClientSession, url: str, dst_dir: str) -> None:
    for i in range(MAX_TRIES):
        try:
            resp = await session.request(method='GET', url=url)
            resp.raise_for_status()
            dst_file = os.path.join(dst_dir, url.rsplit('/', maxsplit=1)[-1])
            async with aiofiles.open(dst_file, 'wb') as f:
                await f.write(await resp.read())
        except Exception as e:
            print('Error: {}'.format(e), flush=False)
            await asyncio.sleep(1)
        else:
            break


async def save_mask(mask: Image, url: str, dst_dir: str) -> None:
    dst_file = os.path.join(dst_dir, url.rsplit('/', maxsplit=1)[-1])
    dst_file = dst_file.replace('.jpg', '.png')
    async with aiofiles.open(dst_file, 'wb') as f:
        byteIO = io.BytesIO()
        mask.save(byteIO, format='PNG')
        byteArr = byteIO.getvalue()
        await f.write(byteArr)


def get_mask_form_anns(anns: List[Any], coco: COCO) -> Image.Image:
    mask = np.array(decode(coco.annToRLE(anns[0])))
    for ann in anns[1:]:
        mask += np.array(decode(coco.annToRLE(ann)))
    mask = (mask > 0).astype(np.uint8)
    return Image.fromarray(mask, mode='L')


def get_urls_and_masks(coco: COCO) -> List[Tuple[str, Image.Image]]:
    cat_ids = coco.getCatIds(catNms=['person'])
    img_ids = coco.getImgIds(catIds=cat_ids)

    urls_masks = []
    for img_id in img_ids:
        annIds = coco.getAnnIds(
            imgIds=img_id, catIds=cat_ids, iscrowd=False)
        anns = coco.loadAnns(annIds)
        img_obj = coco.loadImgs(img_id)[0]

        area = sum((ann['area'] for ann in anns))
        img_area = img_obj['height'] * img_obj['width']
        max_allowed_area = PERSON_MAX_AREA_THRESHOLD * img_area
        min_allowed_area = PERSON_MIN_AREA_THRESHOLD * img_area

        if (len(anns) <= MAX_PPL_ALLOWED and
                min_allowed_area <= area <= max_allowed_area):
            mask = get_mask_form_anns(anns, coco)
            urls_masks.append((img_obj['coco_url'], mask))

    return urls_masks


async def gather_data(urls_and_masks: List[Tuple[str, Image.Image]],
                      img_dst: str, mask_dst: str) -> None:
    img_downloading_tasks = []
    async with ClientSession() as session:
        for url, _ in urls_masks:
            img_downloading_tasks.append(
                download_image(session, url, img_dst))
        await asyncio.gather(*img_downloading_tasks)
    masks_saving_tasks = []
    for url, mask in urls_masks:
        masks_saving_tasks.append(
            save_mask(mask, url, mask_dst))
    await asyncio.gather(*masks_saving_tasks)


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description='Script to get filtered images of ppl from COCO data.')
    parser.add_argument(
        '--anns_path',
        help=('Path to json file with annotations'),
    )
    parser.add_argument(
        '--img_dst',
        help=('Where to copy filtered images'),
    )
    parser.add_argument(
        '--mask_dst',
        help=('Where to save created masks images'),
    )

    args = parser.parse_args()

    try:
        os.makedirs(args.img_dst)
    except FileExistsError:
        print('DST Exists:', args.img_dst)
    else:
        print('DST created', args.img_dst)

    try:
        os.makedirs(args.mask_dst)
    except FileExistsError:
        print('MASK DST exists:', args.mask_dst)
    else:
        print('MASK DST created', args.mask_dst)

    coco = COCO(args.anns_path)

    urls_masks = get_urls_and_masks(coco)
    print('Found: {} images'.format(len(urls_masks)))

    try:
        asyncio.run(gather_data(urls_masks, args.img_dst, args.mask_dst))
    except AttributeError:  # if python 3.6
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            gather_data(urls_masks, args.img_dst, args.mask_dst))
        loop.close()

    print('--- {0:.4f} seconds ---'.format((time.time() - start_time)))
