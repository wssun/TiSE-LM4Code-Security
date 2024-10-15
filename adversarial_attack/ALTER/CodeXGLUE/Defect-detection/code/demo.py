# # pip install -q transformers
# import string
# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
# import torch
# import random
#
# def get_device(gpu_input):
#     device = f'cuda:{gpu_input}' if torch.cuda.is_available() and int(
#         gpu_input) < torch.cuda.device_count() else 'cpu'
#     return device
#
#
# checkpoint = "/home/david/GYF/CodeLlama-7b-Instruct-hf"
# device = get_device("0") # for GPU usage or "cpu" for CPU usage
# config = AutoConfig.from_pretrained(checkpoint)
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# tokenizer.pad_token = tokenizer.eos_token
# model = AutoModelForSequenceClassification.from_pretrained(
#             checkpoint,
#             config=config,
#             torch_dtype=torch.float16,
#             use_safetensors=False).to(device)
#
# # model = AutoModelForCausalLM.from_pretrained(
# #             checkpoint,
# #             config=config,
# #             torch_dtype=torch.float16,
# #             use_safetensors=False).to(device)
#
#
# # 修改函数，用随机英文字母填充字符串
# # codellama最长输入ids字符在5100左右！！！
# def set_question_length(question, length):
#     """ 调整问题字符串到所需长度，超出则截断，不足则用随机英文字母填充 """
#     question = question.strip()
#     if len(question) > length:
#         return question[:length]  # 截断到所需长度
#     while len(question) < length:
#         question += random.choice(string.ascii_letters)  # 使用随机英文字母填充
#     return question
#
# # Example use
# original_question = "pb"
# desired_length = 1100  # Set desired length here
# question = set_question_length(original_question, desired_length)
# # """
# # static int xen_9pfs_connect(struct XenDevice *xendev)  {      i…
# # """
# # """
# # static int read_code_table(CLLCContext *ctx, GetBitContext *gb, VLC *vlc) {     uint8_t symbols[256];     uint8_t bits[256];     uint16_t codes[256];     int num_lens, num_codes, num_codes_sum, prefix;     int i, j, count;     prefix        = 0;     count         = 0;     num_codes_sum = 0;     num_lens = get_bits(gb, 5);     for (i = 0; i < num_lens; i++) {         num_codes      = get_bits(gb, 9);         num_codes_sum += num_codes;         if (num_codes_sum > 256) {             av_log(ctx->avctx, AV_LOG_ERROR,                    "Too many VLCs (%d) to be read.\n", num_codes_sum);         for (j = 0; j < num_codes; j++) {             symbols[count] = get_bits(gb, 8);             bits[count]    = i + 1;             codes[count]   = prefix++;             count++;         if (prefix > (65535 - 256)/2) {         prefix <<= 1;     return ff_init_vlc_sparse(vlc, VLC_BITS, count, bits, 1, 1,                               codes, 2, 2, symbols, 1, 1, 0);
# # """
#
#
# # 结尾不要加</s>
# # I use following template for finetuning and inference:
# #
# # <s>[INST] user_message_1 [/INST] response_1 </s><s>[INST] user_message_2 [/INST] response_2 </s>
# print('Question length:{}'.format(len(question)))
# code = ' '.join(question.split())
# # code_tokens = tokenizer.tokenize(code)[:2048]
# code_tokens = tokenizer.tokenize(code)
# code_tokens = code_tokens
#
# ids = tokenizer.convert_tokens_to_ids(code_tokens)
# print('ids length:{}'.format(len(ids)))
# source_ids = torch.tensor([ids]).to(device)
# # source_ids = tokenizer.encode(question, return_tensors="pt").to(device)
# # inputs = tokenizer([code_tokens], return_tensors='pt', add_special_tokens=False).to(device)
# # outputs1 = model.generate(source_ids, max_new_tokens=10, pad_token_id=tokenizer.pad_token_id,)
# output = model(source_ids)
# logits = output[0]
# prob = torch.softmax(logits, dim=1)
# logits = torch.softmax(logits, dim=1)
# # outputs2 = model.generate(source_ids, max_new_tokens=8)
# # outputs3 = model.generate(source_ids, max_new_tokens=7)
# # yy = outputs1[:,-10:][0]
# # print(yy)
# print("xx")

# pip install -q transformers
import string
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import torch
import random

def get_device(gpu_input):
    device = f'cuda:{gpu_input}' if torch.cuda.is_available() and int(
        gpu_input) < torch.cuda.device_count() else 'cpu'
    return device
checkpoint = "/home/david/GYF/CodeLlama-7b-Instruct-hf"
device = get_device("0")
config = AutoConfig.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            config=config,
            torch_dtype=torch.float16,
            use_safetensors=False).to(device)

question = """
Question :
Code 1 : public static String sha1(String text) throws NoSuchAlgorithmException, UnsupportedEncodingException {
        MessageDigest md;
        md = MessageDigest.getInstance("SHA-1");
        byte[] sha1hash = new byte[40];
        md.update(text.getBytes("iso-8859-1"), 0, text.length());
        sha1hash = md.digest();
        return convertToHex(sha1hash);
    }
Code 2 : public void resolvePlugins() {
        try {
            File cacheDir = XPontusConfigurationConstantsIF.XPONTUS_CACHE_DIR;
            File pluginsFile = new File(cacheDir, "plugins.xml");
            if (!pluginsFile.exists()) {
                URL pluginURL = new URL("http://xpontus.sourceforge.net/snapshot/plugins.xml");
                InputStream is = pluginURL.openStream();
                OutputStream os = FileUtils.openOutputStream(pluginsFile);
                IOUtils.copy(is, os);
                IOUtils.closeQuietly(os);
                IOUtils.closeQuietly(is);
            }
            resolvePlugins(pluginsFile.getAbsolutePath());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
Is there a clone relation between the Code1 and Code2, and respond to YES or NO.
"""

# inputs = tokenizer.encode(question, return_tensors="pt").to(device)
# outputs = model.generate(inputs, max_new_tokens=30, pad_token_id=tokenizer.pad_token_id, do_sample=True)
# tokens = tokenizer.convert_ids_to_tokens(outputs[0])
#
# # 取最后30个tokens
# last_30_tokens = tokens[-30:]
#
# # 将tokens重新组合成字符串形式
# answer = tokenizer.convert_tokens_to_string(last_30_tokens)

input_ids = tokenizer(question, return_tensors="pt").input_ids.to(device)

aa = input_ids.shape[1]

outputs = model.generate(input_ids, max_new_tokens=50)

print(tokenizer.decode(outputs[0][aa:]))

# print(answer)

