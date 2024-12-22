import os
import pickle
import time
from collections import Counter

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.io import ImageReadMode, read_image
from torchvision.models import alexnet, resnet18, resnet50


class MACSADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        img_folder,
        roi_df,
        dict_image_aspect,
        dict_roi_aspect,
        num_img,
        num_roi,
    ):
        self.data = data
        self.ASPECT = [
            "Location",
            "Food",
            "Room",
            "Facilities",
            "Service",
            "Public_area",
        ]

        self.pola_to_num = {"None": 0, "Negative": 1, "Neutral": 2, "Positive": 3}

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224), antialias=True
                ),  # args.crop_size, by default it is set to be 224
                # transforms.RandomHorizontalFlip(),
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self.roi_df = roi_df
        self.img_folder = img_folder
        self.dict_image_aspect = dict_image_aspect
        self.dict_roi_aspect = dict_roi_aspect
        self.num_img = num_img
        self.num_roi = num_roi
        self.tokenizer = tokenizer

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        idx_data = self.data.iloc[idx, :].values

        text = idx_data[0]

        # img = os.path.join(self.img_folder,idx_data[2])
        list_image_aspect = []
        list_roi_aspect = []
        for img_name in idx_data[1][: self.num_img]:
            try:
                list_image_aspect.extend(self.dict_image_aspect[img_name])
            except:
                pass
            try:
                list_roi_aspect.extend(self.dict_roi_aspect[img_name])
            except:
                pass
        list_image_aspect = list(set(list_image_aspect))
        list_roi_aspect = list(set(list_roi_aspect))

        if len(list_image_aspect) == 0:
            list_image_aspect = ["empty"]
        if len(list_roi_aspect) == 0:
            list_roi_aspect = ["empty"]
        # list_image_aspect = list(map(lambda x: x.lower(),list_image_aspect))
        # list_roi_aspect = list(map(lambda x: x.lower(),list_roi_aspect))

        text_img_label = idx_data[3]
        list_aspect, list_polar = [], []
        for asp_pol in text_img_label:
            asp, pol = asp_pol.split("#")
            if "_" in asp:
                asp = "Public area"
            list_aspect.append(asp)
            list_polar.append(pol)

        for asp in self.ASPECT:
            if "_" in asp:
                asp = "Public area"
            if asp not in list_aspect:
                list_aspect.append(asp)
                list_polar.append("None")

        all_input_ids = []
        all_token_types_ids = []
        all_attn_mask = []
        all_label_id = []
        all_added_input_mask = []

        for ix in range(len(self.ASPECT)):
            asp = self.ASPECT[ix]
            if "_" in asp:
                asp = "Public area"

            idx_asp_in_list_asp = list_aspect.index(asp)

            joined_aspect = f" {' , '.join(list_image_aspect)} </s></s>  {' , '.join(list_roi_aspect)}"
            joined_aspect = joined_aspect.lower().replace("_", " ")

            combine_text = f"{asp} </s></s> {text}"
            combine_text = combine_text.lower().replace("_", " ")
            tokens = self.tokenizer(
                combine_text,
                joined_aspect,
                max_length=170,
                truncation="only_first",
                padding="max_length",
                return_token_type_ids=True,
            )

            input_ids = torch.tensor(tokens["input_ids"])
            token_type_ids = torch.tensor(tokens["token_type_ids"])
            attention_mask = torch.tensor(tokens["attention_mask"])
            added_input_mask = torch.tensor([1] * (170 + 49))

            label_id = list_polar[idx_asp_in_list_asp]

            all_input_ids.append(input_ids)
            all_token_types_ids.append(token_type_ids)
            all_attn_mask.append(attention_mask)
            all_added_input_mask.append(added_input_mask)
            all_label_id.append(self.pola_to_num[label_id])

        all_input_ids = torch.stack(all_input_ids, dim=0)
        all_token_types_ids = torch.stack(all_token_types_ids)
        all_attn_mask = torch.stack(all_attn_mask)
        all_added_input_mask = torch.stack(all_added_input_mask)

        all_label_id = torch.tensor(all_label_id)

        list_img_path = idx_data[1]
        # os_list_img_path = [os.path.join(self.img_folder,path) for path in list_img_path]

        list_img_features = []
        global_roi_features = []  # num_img, num_roi, 3, 224, 224
        global_roi_coor = []
        for img_path in list_img_path[: self.num_img]:
            image_os_path = os.path.join(self.img_folder, img_path)
            one_image = read_image(image_os_path, mode=ImageReadMode.RGB)
            img_transform = self.transform(one_image).unsqueeze(0)  # 1, 3, 224, 224
            list_img_features.append(img_transform)

            ##### ROI
            list_roi_img = []  # num_roi, 3, 224, 224
            list_roi_coor = []  # num_roi, 4
            roi_in_img_df = self.roi_df[self.roi_df["file_name"] == img_path][
                : self.num_roi
            ]
            #   print(roi_in_img_df)
            if roi_in_img_df.shape[0] == 0:
                list_roi_img = np.zeros((self.num_roi, 3, 224, 224))

                #   print(len(list_roi_img))
                global_roi_coor.append(np.zeros((self.num_roi, 4)))
                global_roi_features.append(list_roi_img)
                continue

            for i_roi in range(roi_in_img_df.shape[0]):
                x1, x2, y1, y2 = roi_in_img_df.iloc[i_roi, 1 : 4 + 1].values

                roi_in_image = one_image[:, x1:x2, y1:y2]
                roi_transform = self.transform(roi_in_image).numpy()  # 3, 224, 224

                x1, x2, y1, y2 = x1 / 512, x2 / 512, y1 / 512, y2 / 512
                cv = lambda x: np.clip([x], 0.0, 1.0)[0]
                x1 = cv(x1)
                x2 = cv(x2)
                y1 = cv(y1)
                y2 = cv(y2)

                list_roi_coor.append([x1, x2, y1, y2])
                list_roi_img.append(roi_transform)

            #   print("For loop first:", len(list_roi_img))
            if i_roi < self.num_roi:
                for k in range(self.num_roi - i_roi - 1):
                    list_roi_img.append(np.zeros((3, 224, 224)))
                    list_roi_coor.append(np.zeros((4,)))

            global_roi_features.append(list_roi_img)
            global_roi_coor.append(list_roi_coor)

        ### FOR IMAGE
        t_img_features = torch.zeros((self.num_img, 3, 224, 224))
        num_imgs = len(list_img_features)

        if num_imgs >= self.num_img:
            for i in range(self.num_img):
                t_img_features[i, :] = list_img_features[i]
        else:
            for i in range(self.num_img):
                if i < num_imgs:
                    # img_features[(self.max_img_len-num_imgs)+i,:] = imgs[i]
                    t_img_features[i, :] = list_img_features[i]
                else:
                    break

        ### FOR ROI
        global_roi_features = np.asarray(global_roi_features)
        global_roi_coor = np.asarray(global_roi_coor)

        roi_img_features = np.zeros((self.num_img, self.num_roi, 3, 224, 224))
        roi_coors = np.zeros((self.num_img, self.num_roi, 4))

        num_img_roi = len(global_roi_features)

        if num_img_roi >= self.num_img:
            for i in range(self.num_img):
                roi_img_features[i, :] = global_roi_features[i]
                roi_coors[i, :] = global_roi_coor[i]
        else:
            for i in range(self.num_img):
                if i < num_imgs:
                    # img_features[(self.max_img_len-num_imgs)+i,:] = imgs[i]
                    roi_img_features[i, :] = global_roi_features[i]
                    roi_coors[i, :] = global_roi_coor[i, :]
                else:
                    break

        roi_img_features = torch.tensor(roi_img_features)
        roi_coors = torch.tensor(roi_coors)

        return (
            t_img_features,
            roi_img_features,
            roi_coors,
            all_input_ids,
            all_token_types_ids,
            all_attn_mask,
            all_added_input_mask,
            all_label_id,
        )


def load_word_vec(path, word2idx=None):
    fin = open(path, "r", encoding="utf-8", newline="\n", errors="ignore")
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            word_vec[tokens[0]] = np.asarray(tokens[1:], dtype="float32")
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, type):
    embedding_matrix_file_name = "{0}_{1}_embedding_matrix.dat".format(
        str(embed_dim), type
    )
    if os.path.exists(embedding_matrix_file_name):
        print("loading embedding_matrix:", embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, "rb"))
    else:
        print("loading word vectors...")
        embedding_matrix = np.random.rand(
            len(word2idx) + 2, embed_dim
        )  # idx 0 and len(word2idx)+1 are all-zeros
        fname = (
            "../../datasets/GloveData/glove.6B." + str(embed_dim) + "d.txt"
            if embed_dim != 300
            else "../../datasets/ChineseWordVectors/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim"
            + str(embed_dim)
            + ".iter5"
        )
        word_vec = load_word_vec(fname, word2idx=word2idx)
        print("building embedding_matrix:", embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, "wb"))
    return embedding_matrix


class Tokenizer(object):
    def __init__(self, lower=False, max_seq_len=None, max_aspect_len=None):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.max_aspect_len = max_aspect_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    @staticmethod
    def pad_sequence(
        sequence, maxlen, dtype="int64", padding="pre", truncating="pre", value=0.0
    ):
        x = (np.ones(maxlen) * value).astype(dtype)
        if truncating == "pre":
            trunc = sequence[-maxlen:]
        else:
            trunc = sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == "post":
            x[: len(trunc)] = trunc
        else:
            x[-len(trunc) :] = trunc
        return x

    def text_to_sequence(self, text, isaspect=False, reverse=False):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx) + 1
        sequence = [
            self.word2idx[w] if w in self.word2idx else unknownidx for w in words
        ]
        if len(sequence) == 0:
            sequence = [0]
        pad_and_trunc = "post"  # use post padding together with torch.nn.utils.rnn.pack_padded_sequence
        if reverse:
            sequence = sequence[::-1]
        if isaspect:
            return Tokenizer.pad_sequence(
                sequence,
                self.max_aspect_len,
                dtype="int64",
                padding=pad_and_trunc,
                truncating=pad_and_trunc,
            )
        else:
            return Tokenizer.pad_sequence(
                sequence,
                self.max_seq_len,
                dtype="int64",
                padding=pad_and_trunc,
                truncating=pad_and_trunc,
            )


class MIMNDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset="vimacsa",
        embed_dim=100,
        max_seq_len=320,
        max_aspect_len=2,
        max_img_len=5,
        cnn_model_name="resnet50",
    ):
        start = time.time()
        print("Preparing {0} datasets...".format(dataset))

        # Dataset paths
        fname = {
            "zol_cellphone": {
                "train": "./datasets/zolDataset/zol_Train_jieba.txt",
                "dev": "./datasets/zolDataset/zol_Dev_jieba.txt",
                "test": "./datasets/zolDataset/zol_Test_jieba.txt",
            }
        }

        # CNN Model
        cnn_classes = {
            "resnet18": resnet18(pretrained=True),
            "resnet50": resnet50(pretrained=True),
            "alexnet": alexnet(pretrained=True),
        }

        self.cnn_model_name = cnn_model_name
        self.max_img_len = max_img_len
        self.cnn_extractor = torch.nn.Sequential(
            *list(cnn_classes[cnn_model_name].children())[:-1]
        )
        self.transform_img = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        # Tokenizer and Text Processing
        text = self.__read_text__(
            [fname[dataset]["train"], fname[dataset]["dev"], fname[dataset]["test"]]
        )
        tokenizer = Tokenizer(max_seq_len=max_seq_len, max_aspect_len=max_aspect_len)
        tokenizer.fit_on_text(text)

        self.word2idx = tokenizer.word2idx
        self.idx2word = tokenizer.idx2word
        self.embedding_matrix = build_embedding_matrix(
            tokenizer.word2idx, embed_dim, dataset
        )

        # Read datasets
        self.train_data = self.__read_data__(fname[dataset]["train"], tokenizer)
        self.dev_data = self.__read_data__(fname[dataset]["dev"], tokenizer)
        self.test_data = self.__read_data__(fname[dataset]["test"], tokenizer)

        end = time.time()
        m, s = divmod(end - start, 60)
        print("Time to read datasets: %02d:%02d" % (m, s))

    def __getitem__(self, index):
        return self.train_data[index]

    def __len__(self):
        return len(self.train_data)

    @staticmethod
    def __read_text__(fnames):
        text = ""
        for fname in fnames:
            with open(
                fname, "r", encoding="utf-8", newline="\n", errors="ignore"
            ) as fin:
                lines = fin.readlines()
                for i in range(0, len(lines), 4):
                    text_raw = lines[i].strip()
                    text += text_raw + " "
        return text

    def __read_data__(self, fname, tokenizer):
        polarity_dic = {
            "10.0": 8,
            "8.0": 7,
            "6.0": 6,
            "5.0": 5,
            "4.0": 4,
            "3.0": 3,
            "2.0": 2,
            "1.0": 1,
        }
        data_path = fname.split(".txt")[0] + "/"
        os.makedirs(data_path, exist_ok=True)
        data_path = os.path.join(data_path, self.cnn_model_name)
        os.makedirs(data_path, exist_ok=True)

        all_data = []
        with open(fname, "r", encoding="utf-8", newline="\n", errors="ignore") as fin:
            lines = fin.readlines()
            for i in range(0, len(lines), 4):
                fname_i = os.path.join(data_path, f"{int(i / 4)}.pkl")
                if os.path.exists(fname_i):
                    with open(fname_i, "rb") as fpkl:
                        data = pickle.load(fpkl)
                else:
                    print(fname_i)
                    text_raw = lines[i].strip()
                    imgs, num_imgs = self.__read_img__(
                        lines[i + 1].strip()[1:-1].split(",")
                    )
                    aspect = lines[i + 2].strip()
                    polarity = int(polarity_dic[(lines[i + 3].strip())] - 1)
                    text_raw_indices = tokenizer.text_to_sequence(
                        text_raw, isaspect=False
                    )
                    aspect_indices = tokenizer.text_to_sequence(aspect, isaspect=True)
                    data = {
                        "text_raw_indices": text_raw_indices,
                        "imgs": imgs,
                        "num_imgs": num_imgs,
                        "aspect_indices": aspect_indices,
                        "polarity": int(polarity),
                    }
                    with open(fname_i, "wb") as fpkl:
                        pickle.dump(data, fpkl)
                all_data.append(data)
        return all_data

    def __read_img__(self, imgs_path):
        imgs = []
        for img_path in imgs_path:
            img_path = img_path.strip().replace("'", "")
            try:
                img = Image.open(
                    "/home/xunan/code/pytorch/ZOLspider/multidata_zol/img/" + img_path
                ).convert("RGB")
                input_img = self.transform_img(img).unsqueeze(0)
                output = self.cnn_extractor(input_img).squeeze()
                imgs.append(output)
                img.close()
            except Exception as e:
                print(f"Error processing image: {e}")

        embed_dim_img = len(imgs[0]) if imgs else 0
        img_features = torch.zeros(self.max_img_len, embed_dim_img)
        for i in range(min(self.max_img_len, len(imgs))):
            img_features[i, :] = imgs[i]
        return img_features, min(self.max_img_len, len(imgs))

    @staticmethod
    def __data_counter__(fnames):
        jieba_counter = Counter()
        label_counter = Counter()
        lengths_text, lengths_img = [], []
        max_length_text = min_length_text = max_length_img = min_length_img = 0

        for fname in fnames:
            with open(
                fname, "r", encoding="utf-8", newline="\n", errors="ignore"
            ) as fin:
                lines = fin.readlines()
                for i in range(0, len(lines), 4):
                    text_raw = lines[i].strip()
                    imgs = lines[i + 1].strip()[1:-1].split(",")
                    lengths_text.append(len(text_raw))
                    lengths_img.append(len(imgs))
                    jieba_counter.update(text_raw)
                    label_counter.update([lines[i + 3].strip()])

        print(f"Data count: {len(lengths_text)}")
        print(f"Average text length: {np.mean(lengths_text)}")
        print(f"Average img length: {np.mean(lengths_img)}")
        print(f"Label counts: {label_counter}")
