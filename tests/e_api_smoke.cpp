#include "ai_engine.h"

#include <assert.h>
#include <cstring>
#include <iostream>

int main() {
    const uint8_t invalid_bmp[] = {0, 0, 0, 0};
    const char* ocr = OCR_FindMultiText(invalid_bmp, sizeof(invalid_bmp), "A|B", 0.0f, nullptr);
    assert(ocr != nullptr && ocr[0] == '\0');
    const char* yolo = YOLO_InferJson(0, invalid_bmp, sizeof(invalid_bmp), 0.0f);
    assert(yolo != nullptr && yolo[0] == '\0');
    std::cout << "EasyLanguage JSON ABI smoke passed\n";
    return 0;
}
