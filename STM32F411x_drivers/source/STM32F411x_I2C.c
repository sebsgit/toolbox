#include "STM32F411x_I2C.h"
#include "STM32F411x_RCC.h"

#include <string.h>

void ST_I2C_init(ST_I2C_t *i2c)
{
	i2c->baseAdress->CR1 &= ~(1 << 0); // peripheral disable

	// configure slave address
	{
		i2c->baseAdress->OAR1 &= ~(1 << 15); // 7-bit slave addressing
		i2c->baseAdress->OAR1 |= (1 << 14); // bit 14 in OAR1 register should be kept at 1
		i2c->baseAdress->OAR1 &= ~(0x3F << 1); // clear slave address
		i2c->baseAdress->OAR1 |= (i2c->config.slave_address << 1); // set slave address
	}

	// configure speed
	{
		// configure clock frequency, bits 0-5 in CR2
		uint32_t sys_clock_freq = ST_RCC_get_system_clock_frequency();
		sys_clock_freq /= ST_RCC_get_ahb1_prescaler();
		sys_clock_freq /= ST_RCC_get_apb1_prescaler();
		i2c->baseAdress->CR2 &= ~(0x1F);
		i2c->baseAdress->CR2 |= ((sys_clock_freq / (1000U * 1000U)) & 0x1F);

		uint32_t ccr = 0;
		if (i2c->config.mode == ST_I2C_MODE_STD)
		{
			i2c->baseAdress->CCR &= ~(1 << 15);

			// T_low + T_high = 2 * (CCR * T_sys_clock) => [freq domain] => CCR = freq_sys_clock / (2 * freq_i2c)
			ccr = (sys_clock_freq) / (2U * i2c->config.mode);
		}
		else
		{
			i2c->baseAdress->CCR |= (1 << 15);
			if (i2c->config.fm_duty_cycle == ST_I2C_FM_DUTY_CYCLE_2)
			{
				i2c->baseAdress->CCR &= ~(1 << 14);
				ccr = (sys_clock_freq) / (3U * i2c->config.mode);
			}
			else
			{
				i2c->baseAdress->CCR |= (1 << 14);
				ccr = (sys_clock_freq) / (25U * i2c->config.mode);
			}
		}
		// Bits 11:0 CCR[11:0]: Clock control register in Fm/Sm mode (Master mode)
		i2c->baseAdress->CCR &= ~(0x7FF);
		i2c->baseAdress->CCR |= ccr & 0x7FF;

		// configure T-rise register
		uint32_t t_rise;
		if (i2c->config.mode == ST_I2C_MODE_STD)
		{
			t_rise = (sys_clock_freq / (1000U * 1000U)) + 1;
		}
		else
		{
			t_rise = ((sys_clock_freq * 300U) / (1000U * 1000U * 1000U)) + 1;
		}
		i2c->baseAdress->TRISE = t_rise & 0x3F;
	}

	i2c->baseAdress->CR1 |= (1 << 0); // peripheral enable

	// configure ACK
	if (i2c->config.ack_enable)
	{
		i2c->baseAdress->CR1 |= (1 << 10);
	}
	else
	{
		i2c->baseAdress->CR1 &= ~(1 << 10);
	}
}

void ST_I2C_deinit(ST_I2C_t *i2c)
{
	i2c->baseAdress->CR1 |= (1 << 15); // peripheral software reset
}

uint8_t ST_I2C_is_master(ST_I2C_t *i2c)
{
	return i2c->baseAdress->SR2 & (1 << ST_I2C_SR2_MSL);
}

void ST_I2C_Master_send(ST_I2C_t *i2c, const uint8_t slave_addr, const uint8_t *tx_buffer, const size_t data_len)
{
	// clear ACK failure
	i2c->baseAdress->SR1 &= ~(1 << 10);

	// generate start condition
	i2c->baseAdress->CR1 |= (1 << 8);

	// wait for start condition confirmation
	// TODO: add timeout
	while (!(i2c->baseAdress->SR1 & 0x1));

	// set RW bit to "W = 0" and send the slave address
	uint8_t slave_addr_shifted = (slave_addr << 1);
	slave_addr_shifted &= ~(1);
	i2c->baseAdress->DR = slave_addr_shifted;

	// wait for address sent confirmation
	while (!(i2c->baseAdress->SR1 & 0x2))
	{
		if (i2c->baseAdress->SR1 & (1 << 10)) // ACK failure
		{
			//TODO add return codes
			return;
		}
	}

	// clear ADDR flag by reading SR1 and SR2
	uint32_t unused = i2c->baseAdress->SR1;
	unused = i2c->baseAdress->SR2;
	(void)unused;

	// send data
	//TODO: error checking
	for (size_t i = 0; i < data_len; ++i)
	{
		// wait for TxE
		while (!(i2c->baseAdress->SR1 & (1 << ST_I2C_SR1_TXE)));

		// write DR
		i2c->baseAdress->DR = tx_buffer[i];
	}

	// data transmission finalization: wait for TxE=1 and BTF=1
	while (!(i2c->baseAdress->SR1 & (1 << ST_I2C_SR1_BTF)));
	while (!(i2c->baseAdress->SR1 & (1 << ST_I2C_SR1_TXE)));

	// generate stop condition
	i2c->baseAdress->CR1 |= (1 << 9);
}

void ST_I2C_Master_receive(ST_I2C_t *i2c, const uint8_t slave_addr, uint8_t *rx_buffer, const size_t data_len)
{
	//TODO implement according to the transmission state diagram,
	// Reference manual "STM32F411xC/E advanced Arm-based 32-bit MCUs", page 482
}

void ST_I2C_data_write(ST_I2C_t *i2c, const uint8_t data)
{
	i2c->baseAdress->DR = data;
}

uint8_t ST_I2C_data_read(ST_I2C_t *i2c)
{
	return (uint8_t)i2c->baseAdress->DR;
}

static uint32_t ST_I2C_get_ev_irq_no(ST_I2C_t *i2c)
{
	switch ((uint32_t)i2c->baseAdress)
	{
	case ST_I2C1_BASE_ADDRESS:
		return ST_NVIC_I2C1_EV;
	case ST_I2C2_BASE_ADDRESS:
		return ST_NVIC_I2C2_EV;
	case ST_I2C3_BASE_ADDRESS:
		return ST_NVIC_I2C3_EV;
	default:
		break;
	}
	return 0;
}

static uint32_t ST_I2C_get_err_irq_no(ST_I2C_t *i2c)
{
	switch ((uint32_t)i2c->baseAdress)
	{
	case ST_I2C1_BASE_ADDRESS:
		return ST_NVIC_I2C1_ER;
	case ST_I2C2_BASE_ADDRESS:
		return ST_NVIC_I2C2_ER;
	case ST_I2C3_BASE_ADDRESS:
		return ST_NVIC_I2C3_ER;
	default:
		break;
	}
	return 0;
}

void ST_I2C_irq_control(ST_I2C_t *i2c, uint8_t priority, uint8_t enable)
{
	const uint32_t irq_err = ST_I2C_get_err_irq_no(i2c);
	const uint32_t irq_ev = ST_I2C_get_ev_irq_no(i2c);
	if (!irq_err || !irq_ev)
	{
		return;
	}
	ST_NVIC_configure_interrupt(irq_ev, priority, enable);
	ST_NVIC_configure_interrupt(irq_err, priority, enable);
}

void ST_I2C_callback_control(ST_I2C_t *i2c, uint8_t enable)
{
	if (enable)
	{
		i2c->baseAdress->CR2 |= (1 << ST_I2C_CR2_ITERREN);
		i2c->baseAdress->CR2 |= (1 << ST_I2C_CR2_ITEVEN);
		i2c->baseAdress->CR2 |= (1 << ST_I2C_CR2_ITBUFEN);
	}
	else
	{
		i2c->baseAdress->CR2 &= ~(1 << ST_I2C_CR2_ITERREN);
		i2c->baseAdress->CR2 &= ~(1 << ST_I2C_CR2_ITEVEN);
		i2c->baseAdress->CR2 &= ~(1 << ST_I2C_CR2_ITBUFEN);
	}
}

uint8_t ST_I2C_Master_send_IT(ST_I2C_t *i2c, const uint8_t slave_addr, const uint8_t *tx_buffer, const size_t data_len)
{
	const uint8_t busy = i2c->irq.irq_state;
	if (busy == ST_I2C_IRQ_STATE_IDLE)
	{
		i2c->irq.dev_addr = slave_addr;
		i2c->irq.irq_state = ST_I2C_IRQ_STATE_BUSY_TX;
		i2c->irq.tx_buffer = tx_buffer;
		i2c->irq.tx_buff_len = data_len;

		// enable event, error and buffer interrupts
		ST_I2C_callback_control(i2c, 1);

		// generate start condition
		i2c->baseAdress->CR1 |= (1 << 8);

		return ST_I2C_IRQ_STATE_BUSY_TX;
	}
	return busy;
}

uint8_t ST_I2C_Master_receive_IT(ST_I2C_t *i2c, const uint8_t slave_addr, uint8_t *rx_buffer, const size_t data_len)
{
	const uint8_t busy = i2c->irq.irq_state;
	if (busy == ST_I2C_IRQ_STATE_IDLE)
	{
		i2c->irq.dev_addr = slave_addr;
		i2c->irq.irq_state = ST_I2C_IRQ_STATE_BUSY_RX;
		i2c->irq.rx_buffer = rx_buffer;
		i2c->irq.rx_buff_len = data_len;
		i2c->irq.rx_size = data_len;

		// enable event, error and buffer interrupts
		ST_I2C_callback_control(i2c, 1);

		// generate start condition
		i2c->baseAdress->CR1 |= (1 << 8);

		return ST_I2C_IRQ_STATE_BUSY_RX;
	}
	return busy;
}

void ST_I2C_irq_ev_handler(ST_I2C_t *i2c)
{
	const uint8_t i2c_ev_irq_enanbled = i2c->baseAdress->CR2 & (1 << ST_I2C_CR2_ITEVEN);
	const uint8_t i2c_ev_buf_enabled = i2c->baseAdress->CR2 & (1 << ST_I2C_CR2_ITBUFEN);

	if (i2c_ev_irq_enanbled)
	{
		const uint8_t sb_ev = i2c->baseAdress->SR1 & (1 << ST_I2C_SR1_SB);
		const uint8_t addr_ev = i2c->baseAdress->SR1 & (1 << ST_I2C_SR1_ADDR);
		const uint8_t btf_ev = i2c->baseAdress->SR1 & (1 << ST_I2C_SR1_BTF);
		const uint8_t stop_ev = i2c->baseAdress->SR1 & (1 << ST_I2C_SR1_STOPF);

		if (sb_ev)
		{
			// start bit sent, this means we can continue with the "address phase"
			if (i2c->irq.irq_state == ST_I2C_IRQ_STATE_BUSY_RX)
			{
				// address phase for receiver (TODO)
			}
			else if (i2c->irq.irq_state == ST_I2C_IRQ_STATE_BUSY_TX)
			{
				// address phase to transmitter
				// set RW bit to "W = 0" and send the slave address
				uint8_t slave_addr_shifted = (i2c->irq.dev_addr << 1);
				slave_addr_shifted &= ~(1);
				i2c->baseAdress->DR = slave_addr_shifted;
			}
		}

		if (addr_ev)
		{
			// address ACKed, clear ADDR flag by reading SR1 and SR2
			uint32_t unused = i2c->baseAdress->SR1;
			unused = i2c->baseAdress->SR2;
			(void)unused;
		}

		if (btf_ev)
		{
			// data byte transfer finished
			if (i2c->irq.irq_state == ST_I2C_IRQ_STATE_BUSY_RX)
			{
				;//nothing to do
			}
			else if (i2c->irq.irq_state == ST_I2C_IRQ_STATE_BUSY_TX)
			{
				// check if TXE is set and everything is transferred, then close the communication
				if ((i2c->baseAdress->SR1 & (1 << ST_I2C_SR1_TXE)) && (i2c->irq.tx_buff_len == 0))
				{
					// generate stop condition and reset the irq state
					// only if not "repeated start"
					if (!i2c->irq.rep_start)
					{
						i2c->baseAdress->CR1 |= (1 << 9);
					}

					// disable interrupts
					i2c->baseAdress->CR2 &= ~(1 << ST_I2C_CR2_ITEVEN);
					i2c->baseAdress->CR2 &= ~(1 << ST_I2C_CR2_ITBUFEN);

					memset(&i2c->irq, 0, sizeof(i2c->irq));
					i2c->irq.irq_state = ST_I2C_IRQ_STATE_IDLE;

					ST_I2C_App_Event(i2c, ST_I2C_EVENT_TX_COMPL);
				}
			}
		}

		if (stop_ev)
		{
			// stop event received
			// clear the stop flag (by reading SR1 and updating CR1)
			// SR1 is read above, only write to CR1:
			i2c->baseAdress->CR1 |= 0x0;

			ST_I2C_App_Event(i2c, ST_I2C_EVENT_STOP);
		}

		if (i2c_ev_buf_enabled)
		{
			const uint8_t tx_ev = i2c->baseAdress->SR1 & (1 << ST_I2C_SR1_TXE);
			const uint8_t rx_ev = i2c->baseAdress->SR1 & (1 << ST_I2C_SR1_RXNE);

			if (tx_ev)
			{
				if (ST_I2C_is_master(i2c))
				{
					// transmit buffer empty, transfer the next byte
					if ((i2c->irq.tx_buff_len > 0) && (i2c->irq.irq_state == ST_I2C_IRQ_STATE_BUSY_TX))
					{
						i2c->baseAdress->DR = *i2c->irq.tx_buffer;
						++i2c->irq.tx_buffer;
						--i2c->irq.tx_buff_len;
					}
				}
				else
				{
					// slave should transmit data, forward the request to the application
					if (i2c->baseAdress->SR2 & (1 << ST_I2C_SR2_TRA))
					{
						ST_I2C_App_Event(i2c, ST_I2C_EVENT_SLAVE_TRANSMIT);
					}
				}
			}

			if (rx_ev)
			{
				if (ST_I2C_is_master(i2c))
				{
					// receive buffer not empty
					if ((i2c->irq.rx_buff_len > 0) && (i2c->irq.irq_state == ST_I2C_IRQ_STATE_BUSY_RX))
					{
						//TODO master receiver
					}
				}
				else
				{
					if (!(i2c->baseAdress->SR2 & (1 << ST_I2C_SR2_TRA)))
					{
						ST_I2C_App_Event(i2c, ST_I2C_EVENT_SLAVE_RECEIVE);
					}
				}
			}
		}
	}
}

static void ST_I2C_irq_err_check_clear_notify(ST_I2C_t *i2c, uint8_t flag, uint8_t signal_code)
{
	const uint8_t err_flag = i2c->baseAdress->SR1 & (1 << flag);
	if (err_flag)
	{
		i2c->baseAdress->SR1 &= ~(1 << flag);
		ST_I2C_App_Event(i2c, signal_code);
	}
}

void ST_I2C_irq_err_handler(ST_I2C_t *i2c)
{
	const uint8_t i2c_err_irq_enanbled = i2c->baseAdress->CR2 & (1 << ST_I2C_CR2_ITERREN);
	if (i2c_err_irq_enanbled)
	{
		ST_I2C_irq_err_check_clear_notify(i2c, ST_I2C_SR1_BERR, ST_I2C_EVENT_ERR_BUS);
		ST_I2C_irq_err_check_clear_notify(i2c, ST_I2C_SR1_ARLO, ST_I2C_EVENT_ERR_ARBITRATION_LOSS);
		ST_I2C_irq_err_check_clear_notify(i2c, ST_I2C_SR1_AF, ST_I2C_EVENT_ERR_ACK_FAIL);
		ST_I2C_irq_err_check_clear_notify(i2c, ST_I2C_SR1_OVR, ST_I2C_EVENT_ERR_OVERRUN);
		ST_I2C_irq_err_check_clear_notify(i2c, ST_I2C_SR1_PEC, ST_I2C_EVENT_ERR_PEC);
		ST_I2C_irq_err_check_clear_notify(i2c, ST_I2C_SR1_TMOUT, ST_I2C_EVENT_ERR_TIMEOUT);
	}
}

void ST_I2C_App_Event(ST_I2C_t *pSpi, uint8_t e_type)
{
	(void)pSpi;
	(void)e_type;
}
