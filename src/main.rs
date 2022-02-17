use std::path::PathBuf;

use image::RgbImage;
use rand::Rng;

const IMG_SIZE: u32 = 255;

type Point = [u32; 2];

struct ImageHandler {
    path: PathBuf,
    box_loc: (Point, Point),
}

impl ImageHandler {
    fn generate_image(save_path: &str) -> ImageHandler {
        let mut image = RgbImage::new(IMG_SIZE, IMG_SIZE);

        let background_color: [u8; 3] = [generate_rand_number(0, 255) as u8, generate_rand_number(0, 255) as u8, generate_rand_number(0, 255) as u8];
        let foreground_color: [u8; 3] = [generate_rand_number(0, 255) as u8, generate_rand_number(0, 255) as u8, generate_rand_number(0, 255) as u8];

        for x in 0..IMG_SIZE {
            for y in 0..IMG_SIZE {
                image.put_pixel(x, y, background_color.into());
            }
        }

        let pt1: Point = [generate_rand_number(0, IMG_SIZE-2), generate_rand_number(0, IMG_SIZE-2)];
        let pt2: Point = [generate_rand_number(pt1[0] + 1, IMG_SIZE), generate_rand_number(pt1[1]+1, IMG_SIZE)];

        //Pick a color, mark the box as that color
        for x in pt1[0]..pt2[0] {
            for y in pt1[1]..pt2[1] {
                image.put_pixel(x, y, foreground_color.into());
            }
        }

        let path = PathBuf::from(format!("{}/{}-{}.{}-{}.{}.png", save_path, generate_rand_number(0, 10000), pt1[0], pt1[1], pt2[0], pt2[1]));
        image.save(&path).unwrap();

        ImageHandler {
            path,
            box_loc: (pt1, pt2),
        }
    }
}

fn generate_rand_number(min_inclusive: u32, max_inclusive: u32) -> u32 {
    //Could be a good idea to initalise this generater once per thread, is there an overhead associated with doing this...?
    let mut rng = rand::thread_rng();
    rng.gen_range(min_inclusive..max_inclusive+1)
}

fn main() {
    let mut res = String::from("filename,xmin,ymin,xmax,ymax");
    for _ in 0..5_000 {
        let img = ImageHandler::generate_image("training");
        res = format!("{}\n{},{},{},{},{}", res, img.path.file_name().unwrap().to_string_lossy(), img.box_loc.0[0], img.box_loc.0[1], img.box_loc.1[0], img.box_loc.1[1]);
    }
    std::fs::write("training.csv", res).unwrap();

    let mut res = String::from("filename,xmin,ymin,xmax,ymax");
    for _ in 0..500 {
        let img = ImageHandler::generate_image("validation");
        res = format!("{}\n{},{},{},{},{}", res, img.path.file_name().unwrap().to_string_lossy(), img.box_loc.0[0], img.box_loc.0[1], img.box_loc.1[0], img.box_loc.1[1]);
    }
    std::fs::write("validation.csv", res).unwrap();
}
