#ifndef STM32F411X_GPIO_H_
#define STM32F411X_GPIO_H_

#include "STM32F411x_defs.h"

typedef struct
{
	uint8_t pin;
	uint8_t mode;
	uint8_t output_type;
	uint8_t output_speed;
	uint8_t pull_up;
	uint8_t alternate_function;
} ST_GPIO_conf_t;

typedef struct
{
	ST_GPIO_reg_t *portAddr;
	ST_GPIO_conf_t config;
} ST_GPIO_t;

#define ST_GPIO_PIN_MODE_INPUT 		0x0U
#define ST_GPIO_PIN_MODE_OUTPUT 	0x1U
#define ST_GPIO_PIN_MODE_ALTERNATE 	0x2U
#define ST_GPIO_PIN_MODE_ANALOG 	0x3U
#define ST_GPIO_PIN_MODE_IRQ_R		0x4U
#define ST_GPIO_PIN_MODE_IRQ_F		0x5U
#define ST_GPIO_PIN_MODE_IRQ_RF		0x6U

#define ST_GPIO_PIN_OUTPUT_PUSH_PULL 	0x0U
#define ST_GPIO_PIN_OUTPUT_OPEN_DRAIN	0x1U

#define ST_GPIO_PIN_OUT_SPEED_LOW 		0x0U
#define ST_GPIO_PIN_OUT_SPEED_MEDIUM 	0x1U
#define ST_GPIO_PIN_OUT_SPEED_FAST 		0x2U
#define ST_GPIO_PIN_OUT_SPEED_HIGH 		0x3U

#define ST_GPIO_PIN_PULL_UD_NONE		0x0U
#define ST_GPIO_PIN_PULL_UD_UP			0x1U
#define ST_GPIO_PIN_PULL_UD_DOWN		0x2U


extern void ST_GPIO_init(ST_GPIO_t * pGpio);
extern void ST_GPIO_deinit(ST_GPIO_reg_t * pGpioReg);

extern void ST_GPIO_clock_control(ST_GPIO_reg_t *pGpioReg, uint8_t enable);

extern uint8_t ST_GPIO_read_pin(ST_GPIO_reg_t *pGpioReg, uint8_t pin);
extern uint16_t ST_GPIO_read_port(ST_GPIO_reg_t *pGpioReg);

extern void ST_GPIO_write_pin(ST_GPIO_reg_t *pGpio, uint8_t pin, uint8_t value);
extern void ST_GPIO_write_port(ST_GPIO_reg_t *pGpio, uint16_t value);

extern int32_t ST_GPIO_IRQ_get_numer(uint8_t pin);

extern void ST_GPIO_IRQ_control(uint8_t pin, uint8_t priority, uint8_t enable);
extern void ST_GPIO_IRQ_handle(uint8_t pin);

#endif // STM32F411X_GPIO_H_
