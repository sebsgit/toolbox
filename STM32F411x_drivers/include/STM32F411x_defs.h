/*
 * BSD 2-Clause License
 * 
 * Copyright (c) 2024, Sebastian Baginski
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * Definitions for the STM32F411xC/E microcontroller.
 * 
 */


#ifndef STM32F411X_DEFS_H_
#define STM32F411X_DEFS_H_

#include <stdint.h>
#include <stddef.h>

#define ST_SRAM_BASE_ADDRESS	0x20000000U
#define ST_SRAM_SIZE			(128U * 1024U)
#define ST_FLASH_BASE_ADDRESS	0x08000000U
#define ST_FLASH_SIZE			(512U * 1024U)

///
/// NVIC registers
///

#define ST_NVIC_SET_ENABLE_BASE_ADDRESS 0xE000E100U
#define ST_NVIC_SET_ENABLE_0 (ST_NVIC_SET_ENABLE_BASE_ADDRESS + 0x00U)
#define ST_NVIC_SET_ENABLE_1 (ST_NVIC_SET_ENABLE_BASE_ADDRESS + 0x04U)
#define ST_NVIC_SET_ENABLE_2 (ST_NVIC_SET_ENABLE_BASE_ADDRESS + 0x08U)
#define ST_NVIC_SET_ENABLE_3 (ST_NVIC_SET_ENABLE_BASE_ADDRESS + 0x0CU)
#define ST_NVIC_SET_ENABLE_4 (ST_NVIC_SET_ENABLE_BASE_ADDRESS + 0x10U)
#define ST_NVIC_SET_ENABLE_5 (ST_NVIC_SET_ENABLE_BASE_ADDRESS + 0x14U)
#define ST_NVIC_SET_ENABLE_6 (ST_NVIC_SET_ENABLE_BASE_ADDRESS + 0x18U)
#define ST_NVIC_SET_ENABLE_7 (ST_NVIC_SET_ENABLE_BASE_ADDRESS + 0x1CU)

#define ST_NVIC_CLR_ENABLE_BASE_ADDRESS 0XE000E180U
#define ST_NVIC_CLR_ENABLE_0 (ST_NVIC_CLR_ENABLE_BASE_ADDRESS + 0x00U)
#define ST_NVIC_CLR_ENABLE_1 (ST_NVIC_CLR_ENABLE_BASE_ADDRESS + 0x04U)
#define ST_NVIC_CLR_ENABLE_2 (ST_NVIC_CLR_ENABLE_BASE_ADDRESS + 0x08U)
#define ST_NVIC_CLR_ENABLE_3 (ST_NVIC_CLR_ENABLE_BASE_ADDRESS + 0x0CU)
#define ST_NVIC_CLR_ENABLE_4 (ST_NVIC_CLR_ENABLE_BASE_ADDRESS + 0x10U)
#define ST_NVIC_CLR_ENABLE_5 (ST_NVIC_CLR_ENABLE_BASE_ADDRESS + 0x14U)
#define ST_NVIC_CLR_ENABLE_6 (ST_NVIC_CLR_ENABLE_BASE_ADDRESS + 0x18U)
#define ST_NVIC_CLR_ENABLE_7 (ST_NVIC_CLR_ENABLE_BASE_ADDRESS + 0x1CU)

#define ST_NVIC_SET_PEND_BASE_ADDRESS 0XE000E200U
#define ST_NVIC_SET_PEND_0 (ST_NVIC_SET_PEND_BASE_ADDRESS + 0x00U)
#define ST_NVIC_SET_PEND_1 (ST_NVIC_SET_PEND_BASE_ADDRESS + 0x04U)
#define ST_NVIC_SET_PEND_2 (ST_NVIC_SET_PEND_BASE_ADDRESS + 0x08U)
#define ST_NVIC_SET_PEND_3 (ST_NVIC_SET_PEND_BASE_ADDRESS + 0x0CU)
#define ST_NVIC_SET_PEND_4 (ST_NVIC_SET_PEND_BASE_ADDRESS + 0x10U)
#define ST_NVIC_SET_PEND_5 (ST_NVIC_SET_PEND_BASE_ADDRESS + 0x14U)
#define ST_NVIC_SET_PEND_6 (ST_NVIC_SET_PEND_BASE_ADDRESS + 0x18U)
#define ST_NVIC_SET_PEND_7 (ST_NVIC_SET_PEND_BASE_ADDRESS + 0x1CU)

#define ST_NVIC_CLR_PEND_BASE_ADDRESS 0XE000E280U
#define ST_NVIC_CLR_PEND_0 (ST_NVIC_CLR_PEND_BASE_ADDRESS + 0x00U)
#define ST_NVIC_CLR_PEND_1 (ST_NVIC_CLR_PEND_BASE_ADDRESS + 0x04U)
#define ST_NVIC_CLR_PEND_2 (ST_NVIC_CLR_PEND_BASE_ADDRESS + 0x08U)
#define ST_NVIC_CLR_PEND_3 (ST_NVIC_CLR_PEND_BASE_ADDRESS + 0x0CU)
#define ST_NVIC_CLR_PEND_4 (ST_NVIC_CLR_PEND_BASE_ADDRESS + 0x10U)
#define ST_NVIC_CLR_PEND_5 (ST_NVIC_CLR_PEND_BASE_ADDRESS + 0x14U)
#define ST_NVIC_CLR_PEND_6 (ST_NVIC_CLR_PEND_BASE_ADDRESS + 0x18U)
#define ST_NVIC_CLR_PEND_7 (ST_NVIC_CLR_PEND_BASE_ADDRESS + 0x1CU)

#define ST_NVIC_ACTIVE_BASE_ADDRESS 0xE000E300U
#define ST_NVIC_ACTIVE_0 (ST_NVIC_ACTIVE_BASE_ADDRESS + 0x00U)
#define ST_NVIC_ACTIVE_1 (ST_NVIC_ACTIVE_BASE_ADDRESS + 0x04U)
#define ST_NVIC_ACTIVE_2 (ST_NVIC_ACTIVE_BASE_ADDRESS + 0x08U)
#define ST_NVIC_ACTIVE_3 (ST_NVIC_ACTIVE_BASE_ADDRESS + 0x0CU)
#define ST_NVIC_ACTIVE_4 (ST_NVIC_ACTIVE_BASE_ADDRESS + 0x10U)
#define ST_NVIC_ACTIVE_5 (ST_NVIC_ACTIVE_BASE_ADDRESS + 0x14U)
#define ST_NVIC_ACTIVE_6 (ST_NVIC_ACTIVE_BASE_ADDRESS + 0x18U)
#define ST_NVIC_ACTIVE_7 (ST_NVIC_ACTIVE_BASE_ADDRESS + 0x1CU)

#define ST_NVIC_PRIO_BASE_ADDRESS 0xE000E400U
#define ST_NVIC_GET_PRIO_REGISTER(irq_n) (volatile uint32_t*)(ST_NVIC_PRIO_BASE_ADDRESS + 4 * ((irq_n) / 4))

// NVIC IRQ numbers
#define ST_NVIC_IRQ_EXTI_0 		(6)
#define ST_NVIC_IRQ_EXTI_1 		(7)
#define ST_NVIC_IRQ_EXTI_2 		(8)
#define ST_NVIC_IRQ_EXTI_3 		(9)
#define ST_NVIC_IRQ_EXTI_4 		(10)
#define ST_NVIC_IRQ_EXTI9_5 	(23)
#define ST_NVIC_IRQ_EXTI15_10 	(40)


///
/// Peripheral buses
///
#define ST_PERIPHERAL_BUS_BASE_ADDRESS	0x40000000U
#define ST_PERIPHERAL_APB1_BASE_ADDRESS ST_PERIPHERAL_BUS_BASE_ADDRESS
#define ST_PERIPHERAL_APB2_BASE_ADDRESS 0x40010000U
#define ST_PERIPHERAL_AHB1_BASE_ADDRESS 0x40020000U
#define ST_PERIPHERAL_AHB2_BASE_ADDRESS 0x50000000U

///
/// AHB1 bus peripherals
///

#define ST_GPIO_A_BASE_ADDRESS 0x40020000U
#define ST_GPIO_B_BASE_ADDRESS 0x40020400U
#define ST_GPIO_C_BASE_ADDRESS 0x40020800U
#define ST_GPIO_D_BASE_ADDRESS 0x40020C00U
#define ST_GPIO_E_BASE_ADDRESS 0x40021000U
#define ST_GPIO_H_BASE_ADDRESS 0x40021C00U
#define ST_FLASH_INTERFACE_REG_BASE_ADDRESS 0x40023C00U
#define ST_RCC_BASE_ADDRESS 0x40023800U

///
/// APB1 bus peripherals
///

#define ST_SPI2_BASE_ADDRESS 0x40003800U
#define ST_SPI3_BASE_ADDRESS 0x40003C00U
#define ST_I2C1_BASE_ADDRESS 0x40005400U

///
/// APB2 bus peripherals
///

#define ST_SPI5_BASE_ADDRESS 0x40015000U
#define ST_SPI4_BASE_ADDRESS 0x40013400U
#define ST_SPI1_BASE_ADDRESS 0x40013000U
#define ST_USART1_BASE_ADDRESS 0x40011000U
#define ST_USART6_BASE_ADDRESS 0x40011400U

// External interrupt control register
#define ST_EXTI_BASE_ADDRESS 				0x40013C00U
#define ST_EXTI_IRQ_MASK_BASE_ADDRESS		(ST_EXTI_BASE_ADDRESS + 0x00U)
#define ST_EXTI_TRIG_RISE_SEL_BASE_ADDRESS	(ST_EXTI_BASE_ADDRESS + 0x08U)
#define ST_EXTI_TRIG_FALL_SEL_BASE_ADDRESS 	(ST_EXTI_BASE_ADDRESS + 0x0CU)
#define ST_EXTI_PENDING_BASE_ADDRESS 		(ST_EXTI_BASE_ADDRESS + 0x14U)
#define ST_EXTI_GET_CRIDX(pin_num)			( (pin_num) / 4 )
#define ST_EXTI_CR_SEL_PORT_A				(0x0)
#define ST_EXTI_CR_SEL_PORT_B				(0x1)
#define ST_EXTI_CR_SEL_PORT_C				(0x2)
#define ST_EXTI_CR_SEL_PORT_D				(0x3)
#define ST_EXTI_CR_SEL_PORT_E				(0x4)
#define ST_EXTI_CR_SEL_PORT_H				(0x7)

// system configuration register
#define ST_SYSCFG_BASE_ADDRESS 0x40013800U

typedef struct
{
	volatile uint32_t MODE;			// port mode register
	volatile uint32_t OTYPE;		// port output type
	volatile uint32_t OSPEED;		// port output speed
	volatile uint32_t PUPD;			// pull-up, pull-down
	volatile uint32_t ID;			// input data
	volatile uint32_t OD;			// output data
	volatile uint32_t BSR;			// bit set / reset
	volatile uint32_t LCK;			// port configuration lock
	volatile uint32_t AFRL;			// alternate funtion low register
	volatile uint32_t AFRH;			// alternate function high register
} ST_GPIO_reg_t;

typedef struct
{
	volatile uint32_t CR1;
	uint32_t RESERVED0;
	volatile uint32_t SR;
	volatile uint32_t DR;
	volatile uint32_t CRCPR;
	volatile uint32_t RXCRCR;
	volatile uint32_t TXCRCR;
	volatile uint32_t I2SCFGR;
	volatile uint32_t I2SPR;
} ST_SPI_reg_t;

_Static_assert(offsetof(ST_SPI_reg_t, CR1) == 0x00, "");
_Static_assert(offsetof(ST_SPI_reg_t, SR) == 0x08, "");
_Static_assert(offsetof(ST_SPI_reg_t, DR) == 0x0C, "");
_Static_assert(offsetof(ST_SPI_reg_t, CRCPR) == 0x10, "");
_Static_assert(offsetof(ST_SPI_reg_t, RXCRCR) == 0x14, "");
_Static_assert(offsetof(ST_SPI_reg_t, TXCRCR) == 0x18, "");
_Static_assert(offsetof(ST_SPI_reg_t, I2SCFGR) == 0x1C, "");
_Static_assert(offsetof(ST_SPI_reg_t, I2SPR) == 0x20, "");

typedef struct
{
	volatile uint32_t CC;			// clock control
	volatile uint32_t PLLCFG;		// PLL config
	volatile uint32_t CFG;			// clock configuration
	volatile uint32_t CIR;			// clock interrupt
	volatile uint32_t AHB1RST;		// AHB1 peripheral reset
	volatile uint32_t AHB2RST;		// AHB2 peripheral reset
	uint32_t RESERVED1;
	uint32_t RESERVED2;
	volatile uint32_t APB1RST;		// APB1 peripheral reset
	volatile uint32_t APB2RST;		// APB2 peripheral reset
	uint32_t RESERVED3;
	uint32_t RESERVED4;
	volatile uint32_t AHB1EN;		// AHB1 clock enable
	volatile uint32_t AHB2EN;		// AHB2 clock enable
	uint32_t RESERVED5;
	uint32_t RESERVED6;
	volatile uint32_t APB1EN;		// APB1 clock enable	
	volatile uint32_t APB2EN;		// APB2 clock enable
	uint32_t RESERVED7;
	uint32_t RESERVED8;
	volatile uint32_t AHB1LPEN;		// AHB1 clock enable, low power mode
	volatile uint32_t AHB2LPEN;		// AHB2 clock enable, low power mode
	uint32_t RESERVED9;
	uint32_t RESERVED10;
	volatile uint32_t APB1LPEN;		// APB1 clock enable, low power mode
	volatile uint32_t APB2LPEN;		// APB2 clock enable, low power mode
	uint32_t RESERVED11;
	uint32_t RESERVED12;
	volatile uint32_t BDC;			// backup domain control
	volatile uint32_t CS;			// clock control and status
	uint32_t RESERVED13;
	uint32_t RESERVED14;
	volatile uint32_t SSCG;			// spread spectrum clock generation
	volatile uint32_t PLLI2SCFG;	// PLLI2S configuration
	uint32_t RESERVED15;
	volatile uint32_t DCKCFG;		// dedicated clocks configuration
} ST_RCC_reg_t;

_Static_assert (offsetof(ST_RCC_reg_t, CC) == 0x0, "");
_Static_assert (offsetof(ST_RCC_reg_t, DCKCFG) == 0x8C, "");

typedef struct
{
	volatile uint32_t IMR;		// interrupt mask register
	volatile uint32_t EMR;		// event mask register
	volatile uint32_t RTSR;		// rising edge trigger
	volatile uint32_t FTSR;		// falling edge trigger
	volatile uint32_t SWIER;	// software interrupt event
	volatile uint32_t PR;		// interrupt pending
} ST_EXTI_reg_t;

typedef struct
{
	volatile uint32_t MEMREMAP;
	volatile uint32_t PERIPHMODECONF;
	volatile uint32_t EXTICONF[4];
	uint32_t RESERVED[2];
	volatile uint32_t COMPCELL;
} ST_SYSCFG_reg_t;

#define ST_GPIOA 	( (ST_GPIO_reg_t*)(ST_GPIO_A_BASE_ADDRESS) )
#define ST_GPIOB 	( (ST_GPIO_reg_t*)(ST_GPIO_B_BASE_ADDRESS) )
#define ST_GPIOC 	( (ST_GPIO_reg_t*)(ST_GPIO_C_BASE_ADDRESS) )
#define ST_GPIOD 	( (ST_GPIO_reg_t*)(ST_GPIO_D_BASE_ADDRESS) )
#define ST_GPIOE 	( (ST_GPIO_reg_t*)(ST_GPIO_E_BASE_ADDRESS) )
#define ST_GPIOH 	( (ST_GPIO_reg_t*)(ST_GPIO_H_BASE_ADDRESS) )
#define ST_SPI1		( (ST_SPI_reg_t*)(ST_SPI1_BASE_ADDRESS) )
#define ST_SPI2		( (ST_SPI_reg_t*)(ST_SPI2_BASE_ADDRESS) )
#define ST_SPI3		( (ST_SPI_reg_t*)(ST_SPI3_BASE_ADDRESS) )
#define ST_SPI4		( (ST_SPI_reg_t*)(ST_SPI4_BASE_ADDRESS) )
#define ST_SPI5		( (ST_SPI_reg_t*)(ST_SPI5_BASE_ADDRESS) )
#define ST_RCC 		( (ST_RCC_reg_t*)(ST_RCC_BASE_ADDRESS) )
#define ST_EXTI 	( (ST_EXTI_reg_t*)(ST_EXTI_BASE_ADDRESS) )
#define ST_SYSCFG   ( (ST_SYSCFG_reg_t*)(ST_SYSCFG_BASE_ADDRESS) )

//
// GPIO clock API
//

#define ST_GPIOA_CLCK_EN() ( ST_RCC->AHB1EN |= (1 << 0) )
#define ST_GPIOB_CLCK_EN() ( ST_RCC->AHB1EN |= (1 << 1) )
#define ST_GPIOC_CLCK_EN() ( ST_RCC->AHB1EN |= (1 << 2) )
#define ST_GPIOD_CLCK_EN() ( ST_RCC->AHB1EN |= (1 << 3) )
#define ST_GPIOE_CLCK_EN() ( ST_RCC->AHB1EN |= (1 << 4) )
#define ST_GPIOH_CLCK_EN() ( ST_RCC->AHB1EN |= (1 << 7) )
#define ST_GPIOA_CLCK_DI() ( ST_RCC->AHB1EN &= ~(1 << 0) )
#define ST_GPIOB_CLCK_DI() ( ST_RCC->AHB1EN &= ~(1 << 1) )
#define ST_GPIOC_CLCK_DI() ( ST_RCC->AHB1EN &= ~(1 << 2) )
#define ST_GPIOD_CLCK_DI() ( ST_RCC->AHB1EN &= ~(1 << 3) )
#define ST_GPIOE_CLCK_DI() ( ST_RCC->AHB1EN &= ~(1 << 4) )
#define ST_GPIOH_CLCK_DI() ( ST_RCC->AHB1EN &= ~(1 << 7) )

//
// SPI clock API
//
#define ST_SPI1_CLOCK_EN() ( ST_RCC->APB2EN |= (1 << 12) )
#define ST_SPI1_CLOCK_DI() ( ST_RCC->APB2EN &= ~(1 << 12) )
#define ST_SPI2_CLOCK_EN() ( ST_RCC->APB1EN |= (1 << 14) )
#define ST_SPI2_CLOCK_DI() ( ST_RCC->APB1EN &= ~(1 << 14) )
#define ST_SPI3_CLOCK_EN() ( ST_RCC->APB1EN |= (1 << 15) )
#define ST_SPI3_CLOCK_DI() ( ST_RCC->APB1EN &= ~(1 << 15) )
#define ST_SPI4_CLOCK_EN() ( ST_RCC->APB2EN |= (1 << 13) )
#define ST_SPI4_CLOCK_DI() ( ST_RCC->APB2EN &= ~(1 << 13) )
#define ST_SPI5_CLOCK_EN() ( ST_RCC->APB2EN |= (1 << 20) )
#define ST_SPI5_CLOCK_DI() ( ST_RCC->APB2EN &= ~(1 << 20) )

//
// I2C clock API
//

#define ST_I2C1_CLCK_EN() ( ST_RCC->APB1EN |= (1 << 21) )
#define ST_I2C1_CLCK_DI() ( ST_RCC->APB1EN &= ~(1 << 21) )

// Sysconfig clock API
#define ST_SYSCFG_CLOCK_EN() ( ST_RCC->APB2EN |= (1 << 14) )
#define ST_SYSCFG_CLOCK_DI() ( ST_RCC->APB2EN &= ~(1 << 14) )

#endif // STM32F411X_DEFS_H_
