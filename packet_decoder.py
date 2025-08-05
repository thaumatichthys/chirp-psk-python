import numpy as np
from parameters import *



# Structure:
# Preamble 0 (1 byte)
# Preamble 1 / sync (1 byte)
# Body length, in units of bytes (2 bytes, BCH (15, 7))
# Payload (1 - 127 bytes)

test_data = [
    0, 0, 0, 0, 0, 0,  # dummy bits

    0, 1, 0, 1, 0, 1, 0, 1,  # preamble 0
    0, 0, 0, 0, 1, 1, 1, 1,  # preamble 1
    0, 0, 0, 0, 0, 0, 0, 0,  # length MSB
    0, 0, 0, 0, 0, 1, 1, 0,  # length LSB (6)

    0, 1, 0, 0, 1, 0, 0, 0,  # H
    0, 1, 1, 0, 0, 1, 0, 1,  # e
    0, 1, 1, 0, 1, 1, 0, 0,  # l
    0, 1, 1, 0, 1, 1, 0, 0,  # l
    0, 1, 1, 0, 1, 1, 1, 1,  # o
    0, 0, 1, 0, 0, 0, 0, 1,  # !

    0, 0, 0, 0, 0, 0,  # dummy bits
]

class PacketDecoder:
    def __init__(self):
        self.PREAMBLE_0 = [ 0, 1, 0, 1, 0, 1, 0, 1 ]  # 01010101
        self.PREAMBLE_1 = [0, 0, 0, 0, 1, 1, 1, 1]  # 00001111

        self.STATE_PREAMBLE_0 = 0
        self.STATE_PREAMBLE_1 = 1
        self.STATE_LENGTH = 2
        self.STATE_BODY = 3


        self.buffer = np.zeros(8, dtype=int)
        self.write_ptr = 0

        self.state = self.STATE_PREAMBLE_0
        self.bit_counter = 0
        self.byte_counter = 0

        self.length_buf = np.uint16(0)
        self.body_len = 0  # processed length (ECC etc)

        self.data_buf = []



    def compareByte(self, reference):
        # compares buffer to reference (call AFTER incrementing write_ptr)
        idx = self.write_ptr
        output = True
        for asd in range(8):
            output &= (reference[asd] == self.buffer[idx])
            idx = (idx + 1) % 8
        return output

    def getByte(self):
        # reads out the local buffer into an uint8
        local_buf = np.zeros(8, dtype=int)
        idx = self.write_ptr

        for asd in range(8):
            local_buf[asd] = self.buffer[idx]
            idx = (idx + 1) % 8
        return np.packbits(local_buf).astype(np.uint8)[0]

    def decodeByteBuf(self):
        # placeholder
        self.body_len = self.length_buf

    def updateSM(self):
        self.bit_counter += 1
        if self.state == self.STATE_PREAMBLE_0:
            if self.compareByte(self.PREAMBLE_0):
                self.state = self.STATE_PREAMBLE_1
                self.bit_counter = 0
        elif self.state == self.STATE_PREAMBLE_1:
            if self.bit_counter == 8:
                if self.compareByte(self.PREAMBLE_1):
                    self.state = self.STATE_LENGTH  # go to length
                else:
                    self.state = self.STATE_PREAMBLE_0  # go back to beginning
                self.bit_counter = 0
        elif self.state == self.STATE_LENGTH:
            if self.bit_counter == 8:
                self.length_buf |= ((self.getByte().astype(np.uint16) & 0x00FF) << 8)
            elif self.bit_counter == 16:
                self.length_buf |= (self.getByte().astype(np.uint16) & 0x00FF)
                self.bit_counter = 0
                self.byte_counter = 0
                self.state = self.STATE_BODY
                self.decodeByteBuf()
        elif self.state == self.STATE_BODY:
            if self.bit_counter == 8:
                self.bit_counter = 0
                self.data_buf.append(self.getByte())
                self.byte_counter += 1
                if self.byte_counter == self.body_len:
                    self.state = self.STATE_PREAMBLE_1  # preamble 1 so the next one can just be started right away
                    return True
        return False

    def abort(self):
        self.state = self.STATE_PREAMBLE_0
        self.bit_counter = 0
        self.byte_counter = 0
        self.data_buf = []
    def pushValue(self, value):
        self.buffer[self.write_ptr] = value
        self.write_ptr = (self.write_ptr + 1) % 8
        if self.updateSM():
            print(f"TRANSMISSION RECEIVED: >>{''.join(chr(i) for i in self.data_buf)}")

    def encodeData(self, data_bytes):
        output = np.zeros(ACQ_AVERAGES * 2)
        output = np.concat([output, np.array(self.PREAMBLE_0)])
        output = np.concat([output, np.array(self.PREAMBLE_1)])
        length = np.uint16(len(data_bytes))
        len_msb = length & 0xFF00
        len_lsb = length & 0x00FF
        print(f"length bytes = {length}")
        output = np.concatenate([output, np.unpackbits(np.array([len_msb], dtype=np.uint8))])
        output = np.concatenate([output, np.unpackbits(np.array([len_lsb], dtype=np.uint8))])
        for i in range(len(data_bytes)):
            output = np.concat([output, np.unpackbits(np.array([data_bytes[i]], dtype=np.uint8))])
        return output

dut = PacketDecoder()
# hong kong polytechnic university POLYU
for bit in test_data:
    dut.pushValue(bit)
