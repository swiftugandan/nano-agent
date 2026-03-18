use ratatui::style::{Color, Style};

#[derive(Clone, Copy, Debug)]
pub enum Theme {
    Bold,
    Hacker,
}

#[derive(Clone, Copy, Debug)]
pub struct ThemeTokens {
    pub bg: Color,
    pub panel_bg: Color,
    pub border: Color,
    pub title: Color,
    pub text: Color,
    pub dim: Color,
    pub accent: Color,
    pub warn: Color,
    pub err: Color,
    pub ok: Color,
}

impl Theme {
    pub fn tokens(self, no_color: bool) -> ThemeTokens {
        if no_color {
            return ThemeTokens {
                bg: Color::Reset,
                panel_bg: Color::Reset,
                border: Color::Reset,
                title: Color::Reset,
                text: Color::Reset,
                dim: Color::Reset,
                accent: Color::Reset,
                warn: Color::Reset,
                err: Color::Reset,
                ok: Color::Reset,
            };
        }
        match self {
            Theme::Bold => ThemeTokens {
                bg: Color::Rgb(10, 12, 18),
                panel_bg: Color::Rgb(16, 18, 28),
                border: Color::Rgb(92, 243, 255),
                title: Color::Rgb(255, 255, 255),
                text: Color::Rgb(230, 235, 255),
                dim: Color::Rgb(145, 150, 180),
                accent: Color::Rgb(255, 190, 72),
                warn: Color::Rgb(255, 190, 72),
                err: Color::Rgb(255, 86, 106),
                ok: Color::Rgb(75, 255, 161),
            },
            Theme::Hacker => ThemeTokens {
                bg: Color::Rgb(0, 6, 0),
                panel_bg: Color::Rgb(0, 12, 0),
                border: Color::Rgb(0, 80, 0),
                title: Color::Rgb(140, 255, 140),
                text: Color::Rgb(205, 255, 205),
                dim: Color::Rgb(70, 140, 70),
                accent: Color::Rgb(0, 255, 120),
                warn: Color::Rgb(220, 255, 120),
                err: Color::Rgb(255, 70, 70),
                ok: Color::Rgb(0, 255, 120),
            },
        }
    }

    pub fn title_style(self, no_color: bool) -> Style {
        let t = self.tokens(no_color);
        Style::default()
            .fg(t.title)
            .add_modifier(ratatui::style::Modifier::BOLD)
    }
}
